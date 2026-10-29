<?php
// Machine learning (administrators): train habit classifiers on the GPU from the
// site's classifications, keep or delete each version, run a kept version over
// projects, flights or chosen particles, and keep or delete each set of results.
//
// The work is done by ml/habit_ml.py (Keras 3) as a background job, one at a time
// (they share the GPU). Files live in work_dir/ml:
//   models/<id>/model.keras, model.json (report), state.json (kept? name)
//   jobs/<id>.json, <id>.log, and each job's inputs
//   keras/   downloaded pretrained weights
// Results of a run are a "classification set": rows in ml_sets / ml_predictions.
declare(strict_types=1);

function ml_dir(string $sub = ''): string { return work_dir('ml' . ($sub ? "/$sub" : '')); }
function ml_python(): string { return (string)(cfg()['ml_python'] ?? admin_cfg('python')); }
function ml_script(): string { return (string)(cfg()['ml_script'] ?? (__DIR__ . '/../../ml/habit_ml.py')); }
function ml_job_file(string $id): string { return ml_dir('jobs') . '/' . preg_replace('/[^0-9A-Za-z-]/', '', $id) . '.json'; }
function ml_model_dir(string $id): string {
    if (!preg_match('/^[0-9A-Za-z-]{6,40}$/', $id)) fail(400, 'Bad model id');
    return ml_dir('models') . "/$id";
}
function ml_env(): string {
    // the web server's own HOME isn't writable; keep Keras's downloads with the models
    return 'env HOME=' . escapeshellarg(ml_dir()) . ' KERAS_HOME=' . escapeshellarg(ml_dir('keras'))
         . ' TF_FORCE_GPU_ALLOW_GROWTH=true TF_CPP_MIN_LOG_LEVEL=2 ';
}
function ml_cmd(string $what, string $jobJson): string {
    return ml_env() . escapeshellarg(ml_python()) . ' -u ' . escapeshellarg(ml_script()) . ' ' . $what . ' ' . escapeshellarg($jobJson);
}

function ml_job_status(array $j): array {
    $tail = '';
    if (is_file($j['log'] ?? '')) {
        $fh = fopen($j['log'], 'r');
        fseek($fh, max(0, filesize($j['log']) - 20000));
        $tail = (string)stream_get_contents($fh);
        fclose($fh);
    }
    if ($j['status'] === 'running') {
        if (preg_match('/@exit (\d+)\s*$/', $tail, $m)) {
            $j['status'] = $m[1] === '0' ? 'done' : 'failed';
            $j['finished'] = gmdate('c');
            ml_job_finished($j);
            write_json_file(ml_job_file($j['id']), $j);
        } elseif (!empty($j['pid']) && !file_exists('/proc/' . (int)$j['pid'])) {
            $j['status'] = 'failed'; $j['finished'] = gmdate('c'); $j['error'] = 'The job stopped without finishing';
            ml_job_finished($j);
            write_json_file(ml_job_file($j['id']), $j);
        }
    }
    $prog = null;
    if (preg_match_all('/@progress (\{.*\})/', $tail, $mm)) $prog = json_decode(end($mm[1]), true);
    // TensorFlow's start-up chatter isn't useful on the page
    $noise = '/^(I0000|W0000|E0000|WARNING: All log messages|To enable the following|.*oneDNN custom operations|.*cpu_feature_guard)/';
    $lines = array_values(array_filter(explode("\n", $tail), fn($l) => $l !== '' && $l[0] !== '@' && !preg_match($noise, $l)));
    $out = $j;
    $out['progress'] = $prog;
    $out['log_tail'] = implode("\n", array_slice($lines, -16));
    unset($out['log'], $out['pid'], $out['cmd']);
    return $out;
}

/** After a job: record a trained model, or mark a classification set failed. */
function ml_job_finished(array $j): void {
    if ($j['kind'] === 'train') {
        $dir = ml_dir('models') . '/' . $j['model'];
        if ($j['status'] === 'done' && is_file("$dir/model.json")) {
            $st = read_json_file("$dir/state.json") ?: [];
            $st['status'] = 'trained';
            write_json_file("$dir/state.json", $st);
        } elseif (is_dir($dir)) {
            rrmdir($dir);                    // nothing worth keeping from a failed run
        }
    } elseif ($j['kind'] === 'predict' && $j['status'] !== 'done') {
        db()->prepare("UPDATE ml_sets SET status = 'failed' WHERE id = ? AND status <> 'ready'")->execute([$j['set']]);
    }
}

function ml_pump(): void {
    if (!exec_available()) return;
    $lock = fopen(ml_dir() . '/jobs.lock', 'c');
    if (!$lock || !flock($lock, LOCK_EX)) return;
    $jobs = [];
    foreach (glob(ml_dir('jobs') . '/*.json') ?: [] as $f) { $j = read_json_file($f); if ($j && isset($j['kind'])) $jobs[] = $j; }
    usort($jobs, fn($a, $b) => strcmp($a['id'], $b['id']));
    $busy = false;
    foreach ($jobs as $j) if ($j['status'] === 'running' && ml_job_status($j)['status'] === 'running') $busy = true;
    if (!$busy) {
        foreach ($jobs as $j) {
            if ($j['status'] !== 'queued') continue;
            $inner = '(' . $j['cmd'] . ') >> ' . escapeshellarg($j['log']) . ' 2>&1; echo "@exit $?" >> ' . escapeshellarg($j['log']);
            file_put_contents($j['log'], '$ ' . $j['cmd'] . "\n", FILE_APPEND);
            $j['pid'] = (int)trim((string)shell_exec('nohup sh -c ' . escapeshellarg($inner) . ' > /dev/null 2>&1 & echo $!'));
            $j['status'] = 'running';
            $j['started'] = gmdate('c');
            write_json_file(ml_job_file($j['id']), $j);
            break;                           // one at a time: they share the GPU
        }
    }
    flock($lock, LOCK_UN);
    fclose($lock);
}

function ml_new_job(string $kind, array $extra, string $cmd, string $user): array {
    $id = gmdate('Ymd-His') . '-' . bin2hex(random_bytes(3));
    $j = array_merge(['id' => $id, 'kind' => $kind, 'status' => 'queued', 'created' => gmdate('c'), 'user' => $user,
                      'log' => ml_dir('jobs') . "/$id.log"], $extra);
    $j['cmd'] = str_replace('@JOB@', $id, $cmd);
    write_json_file(ml_job_file($id), $j);
    ml_pump();
    return ml_job_status(read_json_file(ml_job_file($id)));
}

/** The riming degrees (a flag group: none, lightly, moderately) and the other flags a model can learn. */
function ml_flag_setup(): array {
    $sch = scheme();
    $flags = $sch['flags'] ?? [];
    $groups = array_values(array_unique(array_filter(array_map(fn($f) => $f['group'] ?? '', $flags))));
    $g = in_array('riming', $groups, true) ? 'riming' : ($groups[0] ?? null);
    $levels = ['none'];
    $list = [];
    foreach ($flags as $f) {
        if (!empty($f['group'])) { if ($f['group'] === $g) $levels[] = $f['id']; continue; }
        if ($f['id'] === 'unsure') continue;          // about the person, not the particle
        $list[] = $f['id'];
    }
    return [$levels, $list];
}

/**
 * One answer per particle from people's classifications: the most common class, if at
 * least $minPeople classified it and at least $minAgree of them agree; then, among the
 * people who gave that class, the most common riming degree and the flags at least half
 * of them gave. Returns [pid => [label, flight_id, riming level (0 = none), [flags]]].
 */
function ml_consensus(array $flights, int $minPeople, float $minAgree, bool $withFlags = false): array {
    $where = ''; $args = [];
    if ($flights) { $where = 'WHERE flight_id IN (' . implode(',', array_fill(0, count($flights), '?')) . ')'; $args = $flights; }
    $st = db()->prepare("SELECT pid, flight_id, label, flags FROM labels $where");
    $st->execute($args);
    $by = [];
    foreach ($st as $r) $by[$r['pid']][] = [$r['label'], $r['flight_id'], $withFlags ? (string)$r['flags'] : ''];
    [$levels, $list] = $withFlags ? ml_flag_setup() : [['none'], []];
    $out = [];
    foreach ($by as $pid => $votes) {
        $t = count($votes);
        $n = [];
        foreach ($votes as $v) $n[$v[0]] = ($n[$v[0]] ?? 0) + 1;
        arsort($n);
        $top = array_key_first($n);
        $vals = array_values($n);
        if ($t < $minPeople || $vals[0] / $t < $minAgree) continue;
        if (count($vals) > 1 && $vals[1] === $vals[0]) continue;     // a tie: no answer
        $rim = 0; $fl = [];
        if ($withFlags) {
            $same = array_values(array_filter($votes, fn($v) => $v[0] === $top));
            $rc = array_fill(0, count($levels), 0); $fc = [];
            foreach ($same as $v) {
                $fs = $v[2] === '' ? [] : explode(',', $v[2]);
                $lv = 0;
                foreach ($fs as $f) { $k = array_search($f, $levels, true); if ($k) $lv = max($lv, (int)$k); if (in_array($f, $list, true)) $fc[$f] = ($fc[$f] ?? 0) + 1; }
                $rc[$lv]++;
            }
            $rim = (int)array_search(max($rc), $rc, true);                // the most common degree (the lower one on a tie)
            foreach ($fc as $f => $k) if ($k * 2 >= count($same)) $fl[] = $f;
        }
        $out[$pid] = [$top, $votes[0][1], $rim, $fl];
    }
    return $out;
}

function ml_models_list(): array {
    $out = [];
    foreach (glob(ml_dir('models') . '/*/state.json') ?: [] as $sf) {
        $dir = dirname($sf);
        $st = read_json_file($sf) ?: [];
        $info = read_json_file("$dir/model.json");
        $m = $info['metrics'] ?? [];
        $out[] = ['id' => basename($dir), 'name' => $st['name'] ?? '', 'kept' => !empty($st['kept']), 'status' => $st['status'] ?? 'training',
                  'created' => $info['created'] ?? ($st['created'] ?? null), 'by' => $st['by'] ?? null,
                  'classes' => $info['classes'] ?? [], 'accuracy' => $m['accuracy'] ?? null, 'top2' => $m['top2'] ?? null,
                  'macro_f1' => $m['macro_f1'] ?? null, 'n_train' => $info['n_train'] ?? null, 'n_test' => $info['n_test'] ?? null,
                  'flights' => $st['flights'] ?? [], 'options' => $st['options'] ?? [], 'split' => $info['split'] ?? null,
                  'pretrained' => $info['pretrained'] ?? null, 'img_size' => $info['img_size'] ?? null,
                  'prep' => $info['prep'] ?? null, 'seen' => is_file("$dir/seen.png")];
    }
    usort($out, fn($a, $b) => strcmp((string)$b['id'], (string)$a['id']));
    return $out;
}

function ml_action(string $action, array $u): void {
    ensure_schema();
    switch ($action) {
    case 'admin_ml_check':
        $code = 0; $o = [];
        exec(ml_env() . escapeshellarg(ml_python()) . ' ' . escapeshellarg(ml_script()) . ' check 2>&1', $o, $code);
        $last = json_decode((string)end($o), true);
        out(['python' => ml_python(), 'script' => ml_script(), 'script_found' => is_file(ml_script()),
             'result' => is_array($last) ? $last : null, 'output' => $code === 0 && is_array($last) ? '' : implode("\n", array_slice($o, -12)),
             'gpu_devices' => array_values(array_map('basename', glob('/dev/nvidia[0-9]*') ?: []))]);

    case 'admin_ml_summary':
        // how many particles each flight could give for training, per class
        $minPeople = max(1, (int)($_GET['min_people'] ?? 1));
        $minAgree = max(0.0, min(1.0, (float)($_GET['min_agree'] ?? 0.66)));
        $byFlight = [];
        foreach (ml_consensus([], $minPeople, $minAgree) as [$label, $fid, , ]) {
            $byFlight[$fid][$label] = ($byFlight[$fid][$label] ?? 0) + 1;
        }
        out(['flights' => (object)$byFlight]);

    case 'admin_ml_train':
        $b = require_post();
        $flights = array_values(array_filter((array)($b['flights'] ?? []), fn($f) => is_string($f) && preg_match('/^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$/', $f)));
        if (!$flights) fail(400, 'Choose some flights to train on');
        $o = (array)($b['options'] ?? []);
        $opts = [
            'img_size' => in_array((int)($o['img_size'] ?? 160), [96, 128, 160, 192, 224], true) ? (int)$o['img_size'] : 160,
            'epochs' => max(1, min(200, (int)($o['epochs'] ?? 20))),
            'batch' => max(8, min(512, (int)($o['batch'] ?? 64))),
            'min_examples' => max(2, min(10000, (int)($o['min_examples'] ?? 20))),
            'val_fraction' => max(0.05, min(0.5, (float)($o['val_fraction'] ?? 0.2))),
            'base' => in_array($o['base'] ?? '', ['EfficientNetV2B0', 'EfficientNetV2B1', 'EfficientNetV2S'], true) ? $o['base'] : 'EfficientNetV2B0',
            'seed' => (int)($o['seed'] ?? 1),
            'weights' => empty($o['from_scratch']) ? 'imagenet' : null,
            'orient' => !empty($o['orient']),
            'scale' => ($o['scale'] ?? '') === 'native' ? 'native' : 'fit',
        ];
        $minPeople = max(1, (int)($o['min_people'] ?? 1));
        $minAgree = max(0.0, min(1.0, (float)($o['min_agree'] ?? 0.66)));
        $want = array_values(array_filter((array)($o['classes'] ?? []), 'is_string'));
        $labels = ml_consensus($flights, $minPeople, $minAgree, true);
        [$levels, $flagList] = ml_flag_setup();
        $id = gmdate('Ymd-His') . '-' . bin2hex(random_bytes(2));
        $dir = ml_dir('models') . "/$id";
        @mkdir($dir, 0775, true);
        $csv = "$dir/labels.csv";
        $fh = fopen($csv, 'w');
        fputcsv($fh, ['pid', 'label', 'riming', 'flags']);
        $n = 0;
        foreach ($labels as $pid => [$label, $fid, $rim, $fl]) {
            if ($want && !in_array($label, $want, true)) continue;
            fputcsv($fh, [$pid, $label, $rim, implode(';', $fl)]); $n++;
        }
        fclose($fh);
        if ($n < 20) { rrmdir($dir); fail(400, "Only $n classified particles match: classify more, or loosen the settings"); }
        $sch = scheme();
        $job = ['data_dir' => data_dir(), 'labels_csv' => $csv, 'out_dir' => $dir,
                'classes' => array_column($sch['classes'] ?? [], 'id'), 'options' => $opts,
                'riming_levels' => count($levels) > 1 ? $levels : [], 'flag_list' => $flagList];
        write_json_file("$dir/train.json", $job);
        write_json_file("$dir/state.json", ['status' => 'training', 'kept' => false, 'name' => trim((string)($b['name'] ?? '')),
            'by' => $u['username'], 'created' => gmdate('c'), 'flights' => $flights,
            'options' => $opts + ['min_people' => $minPeople, 'min_agree' => $minAgree, 'classes' => $want], 'n_labels' => $n]);
        out(['job' => ml_new_job('train', ['model' => $id, 'n_labels' => $n, 'flights' => count($flights)], ml_cmd('train', "$dir/train.json"), $u['username'])]);

    case 'admin_ml_jobs':
        ml_pump();
        $jobs = [];
        foreach (glob(ml_dir('jobs') . '/*.json') ?: [] as $f) { $j = read_json_file($f); if ($j && isset($j['kind'])) $jobs[] = $j; }
        usort($jobs, fn($a, $b) => strcmp($b['id'], $a['id']));
        out(['jobs' => array_map('ml_job_status', array_slice($jobs, 0, 12))]);

    case 'admin_ml_cancel':
        $b = require_post();
        $f = ml_job_file((string)($b['id'] ?? ''));
        $j = read_json_file($f);
        if (!$j) fail(404, 'No such job');
        if ($j['status'] === 'running' && !empty($j['pid'])) {
            $pid = (int)$j['pid'];
            shell_exec("pkill -TERM -P $pid 2>/dev/null; sleep 1; pkill -TERM -P $pid 2>/dev/null; kill -TERM $pid 2>/dev/null");
        }
        if (in_array($j['status'], ['running', 'queued'], true)) {
            $j['status'] = 'cancelled'; $j['finished'] = gmdate('c');
            ml_job_finished($j);
            write_json_file($f, $j);
        }
        ml_pump();
        out(['job' => ml_job_status($j)]);

    case 'admin_ml_models':
        ml_pump();                                  // notices a training run that has just finished
        out(['models' => ml_models_list()]);

    case 'admin_ml_model':
        $dir = ml_model_dir((string)($_GET['id'] ?? ''));
        $info = read_json_file("$dir/model.json");
        if (!$info) fail(404, 'No report for that model');
        $info['state'] = read_json_file("$dir/state.json");
        out(['model' => $info]);

    case 'admin_ml_seen':                                   // the sample sheet: particles as the model sees them
        $f = ml_model_dir((string)($_GET['id'] ?? '')) . '/seen.png';
        if (!is_file($f)) fail(404, 'No sample sheet for that model');
        header('Content-Type: image/png');
        header('Cache-Control: private, max-age=86400');
        header('Content-Length: ' . filesize($f));
        readfile($f);
        exit;

    case 'admin_ml_model_keep':
        $b = require_post();
        $dir = ml_model_dir((string)($b['id'] ?? ''));
        $st = read_json_file("$dir/state.json");
        if (!$st || ($st['status'] ?? '') !== 'trained') fail(404, 'No finished model with that id');
        $st['kept'] = !isset($b['kept']) || !empty($b['kept']);
        if (isset($b['name'])) $st['name'] = mb_substr(trim((string)$b['name']), 0, 80);
        write_json_file("$dir/state.json", $st);
        out(['models' => ml_models_list()]);

    case 'admin_ml_model_delete':
        $b = require_post();
        $dir = ml_model_dir((string)($b['id'] ?? ''));
        if (!is_dir($dir)) fail(404, 'No such model');
        foreach (glob(ml_dir('jobs') . '/*.json') ?: [] as $f) {
            $j = read_json_file($f);
            if ($j && ($j['model'] ?? '') === basename($dir) && in_array($j['status'], ['running', 'queued'], true)) fail(409, 'That model is still in use by a job');
        }
        rrmdir($dir);
        out(['models' => ml_models_list()]);

    case 'admin_ml_predict':
        // classify projects/flights, or chosen particles, with a model: makes a classification set
        $b = require_post();
        $mid = (string)($b['model'] ?? '');
        $dir = ml_model_dir($mid);
        $st = read_json_file("$dir/state.json");
        if (!$st || ($st['status'] ?? '') !== 'trained') fail(404, 'No finished model with that id');
        $flights = array_values(array_filter((array)($b['flights'] ?? []), fn($f) => is_string($f) && preg_match('/^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$/', $f)));
        $pids = array_values(array_filter((array)($b['pids'] ?? []), fn($p) => is_string($p) && preg_match(PID_RE, $p)));
        if (!$flights && !$pids) fail(400, 'Choose flights or particles to classify');
        if (count($pids) > 100000) fail(400, 'Too many particles; choose flights instead');
        $name = mb_substr(trim((string)($b['name'] ?? '')), 0, 120)
            ?: (($st['name'] ?: "Model $mid") . ': ' . ($pids && !$flights ? count($pids) . ' particles' : count($flights) . ' flight' . (count($flights) === 1 ? '' : 's')));
        $scope = ['flights' => $flights, 'n_pids' => count($pids)];
        db()->prepare('INSERT INTO ml_sets (model_id, model_name, name, scope, n, status, created_at, created_by) VALUES (?,?,?,?,0,?,?,?)')
            ->execute([$mid, $st['name'] ?: $mid, $name, json_encode($scope), 'running', now_sql(), $u['username']]);
        $sid = (int)db()->lastInsertId();
        $jd = ml_dir('jobs');
        $csv = "$jd/set-$sid.csv";
        $pj = "$jd/set-$sid.json";
        write_json_file($pj, ['data_dir' => data_dir(), 'model_dir' => $dir, 'out_csv' => $csv, 'flights' => $flights, 'pids' => $pids]);
        $php = escapeshellarg((string)(cfg()['php_cli'] ?? 'php'));
        $cmd = ml_cmd('predict', $pj) . ' && ' . $php . ' ' . escapeshellarg(__DIR__ . '/tools/ml_load.php') . " $sid " . escapeshellarg($csv);
        out(['job' => ml_new_job('predict', ['model' => $mid, 'set' => $sid, 'name' => $name], $cmd, $u['username']), 'set' => $sid]);

    case 'admin_ml_sets':
        ml_pump();                                  // also notices finished jobs
        $rows = db()->query('SELECT id, model_id, model_name, name, scope, n, status, created_at, created_by FROM ml_sets ORDER BY id DESC')->fetchAll();
        $fq = db()->prepare('SELECT DISTINCT flight_id FROM ml_predictions WHERE set_id = ?');
        foreach ($rows as &$r) {
            $r['id'] = (int)$r['id']; $r['n'] = (int)$r['n']; $r['scope'] = json_decode((string)$r['scope'], true);
            $r['flights'] = $r['scope']['flights'] ?? [];
            if (!$r['flights'] && $r['status'] === 'ready') { $fq->execute([$r['id']]); $r['flights'] = $fq->fetchAll(PDO::FETCH_COLUMN); }
        }
        unset($r);
        out(['sets' => $rows]);

    case 'admin_ml_set_delete':
        $b = require_post();
        $sid = (int)($b['id'] ?? 0);
        db()->prepare('DELETE FROM ml_predictions WHERE set_id = ?')->execute([$sid]);
        db()->prepare('DELETE FROM ml_sets WHERE id = ?')->execute([$sid]);
        foreach (["set-$sid.csv", "set-$sid.json"] as $f) @unlink(ml_dir('jobs') . "/$f");
        out(['ok' => true]);

    case 'admin_ml_set_rename':
        $b = require_post();
        db()->prepare('UPDATE ml_sets SET name = ? WHERE id = ?')->execute([mb_substr(trim((string)($b['name'] ?? '')), 0, 120), (int)($b['id'] ?? 0)]);
        out(['ok' => true]);

    case 'admin_ml_set_csv':
        $sid = (int)($_GET['id'] ?? 0);
        $st = db()->prepare('SELECT name FROM ml_sets WHERE id = ?');
        $st->execute([$sid]);
        $name = $st->fetchColumn();
        if ($name === false) fail(404, 'No such set');
        @set_time_limit(0);
        if (!is_sqlite()) db()->setAttribute(PDO::MYSQL_ATTR_USE_BUFFERED_QUERY, false);   // millions of rows: don't hold them all
        header('Content-Type: text/csv; charset=utf-8');
        header('Content-Disposition: attachment; filename="cpi-model-set-' . $sid . '.csv"');
        $fh = fopen('php://output', 'w');
        fputcsv($fh, ['pid', 'flight', 'mat_file', 'index', 'label', 'prob', 'label2', 'prob2', 'riming', 'riming_prob', 'flags']);
        $q = db()->prepare('SELECT pid, flight_id, label, prob, label2, prob2, riming, riming_prob, flags FROM ml_predictions WHERE set_id = ? ORDER BY pid');
        $q->execute([$sid]);
        foreach ($q as $r) {
            preg_match(PID_RE, $r['pid'], $m);
            fputcsv($fh, [$r['pid'], $r['flight_id'], $m[2] ?? '', $m[3] ?? '', $r['label'], $r['prob'], $r['label2'], $r['prob2'], $r['riming'], $r['riming_prob'], $r['flags']]);
        }
        exit;

    case 'admin_ml_pred_flight':
        // one flight's results as text, streamed row by row so even millions of particles need
        // little memory: lines "stem:index<TAB>label<TAB>prob<TAB>label2<TAB>prob2<TAB>riming<TAB>flags"
        $sid = (int)($_GET['set'] ?? 0);
        $fid = (string)($_GET['flight'] ?? '');
        if (!preg_match('/^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$/', $fid)) fail(400, 'Bad flight id');
        if (session_status() === PHP_SESSION_ACTIVE) session_write_close();   // let the next flight's request run alongside
        @set_time_limit(0);
        $pdo = db();
        if (!is_sqlite()) $pdo->setAttribute(PDO::MYSQL_ATTR_USE_BUFFERED_QUERY, false);
        $q = $pdo->prepare('SELECT pid, label, prob, label2, prob2, riming, flags FROM ml_predictions WHERE set_id = ? AND flight_id = ?');
        $q->execute([$sid, $fid]);
        header('Content-Type: text/plain; charset=utf-8');
        header('Cache-Control: no-store');
        if (extension_loaded('zlib') && !ini_get('zlib.output_compression') && str_contains($_SERVER['HTTP_ACCEPT_ENCODING'] ?? '', 'gzip')) ob_start('ob_gzhandler', 1 << 20);
        $cut = strlen($fid) + 1;
        $buf = '';
        while (($r = $q->fetch(PDO::FETCH_NUM)) !== false) {
            $buf .= substr($r[0], $cut) . "\t" . $r[1] . "\t" . round((float)$r[2], 4) . "\t" . ($r[3] ?? '') . "\t"
                  . ($r[4] === null ? '' : round((float)$r[4], 4)) . "\t" . ($r[5] ?? '') . "\t" . ($r[6] ?? '') . "\n";
            if (strlen($buf) > 65536) { echo $buf; $buf = ''; }
        }
        echo $buf;
        exit;

    case 'admin_ml_predictions':
        // a set's results for the flights on the Classify page: {pid: [label, prob, label2, prob2]}
        $sid = (int)($_GET['set'] ?? 0);
        $fl = array_values(array_filter(explode(',', (string)($_GET['flights'] ?? '')), fn($f) => preg_match('/^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$/', $f)));
        $out = [];
        if ($fl) {
            $q = db()->prepare('SELECT pid, label, prob, label2, prob2, riming, flags FROM ml_predictions WHERE set_id = ? AND flight_id IN (' . implode(',', array_fill(0, count($fl), '?')) . ')');
            $q->execute(array_merge([$sid], $fl));
            foreach ($q as $r) $out[$r['pid']] = [$r['label'], round((float)$r['prob'], 4), $r['label2'], $r['prob2'] === null ? null : round((float)$r['prob2'], 4),
                                                  $r['riming'] ?: null, $r['flags'] ?: null];
        }
        out(['predictions' => (object)$out]);

    default:
        fail(404, 'Unknown machine learning action');
    }
}
