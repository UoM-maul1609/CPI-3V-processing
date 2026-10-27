<?php
// Admin API: add flights from MAT files that live on the server, and run the
// exporter in the background. Only usernames listed in config 'admins' can
// call these. Every path the browser sends is checked to lie inside raw_root.
declare(strict_types=1);

function admin_cfg(string $k, $default = null) {
    $c = cfg();
    $defaults = [
        'raw_root' => '',
        'work_dir' => '',
        'data_dir' => __DIR__ . '/../data',
        'python'   => 'python3',
        'exporter' => __DIR__ . '/../../exporter/cpi_export.py',
        'max_jobs' => 1,
    ];
    return $c[$k] ?? $defaults[$k] ?? $default;
}

function require_admin(): array {
    $u = require_user();
    if (!$u['is_admin']) fail(403, 'Only administrators can do this');
    return $u;
}

function raw_root(): string {
    $r = admin_cfg('raw_root');
    if (!$r) fail(500, "Set 'raw_root' in api/config.php to the folder that holds your MAT files");
    $real = realpath($r);
    if ($real === false || !is_dir($real)) fail(500, "raw_root ($r) does not exist or is not a folder");
    return $real;
}

function work_dir(string $sub = ''): string {
    $w = admin_cfg('work_dir');
    if (!$w) fail(500, "Set 'work_dir' in api/config.php to a folder the web server can write to");
    $dir = rtrim($w, '/') . ($sub ? "/$sub" : '');
    if (!is_dir($dir) && !@mkdir($dir, 0775, true)) fail(500, "Can't create $dir - check permissions");
    return realpath($dir);
}

function data_dir(): string {
    $d = admin_cfg('data_dir');
    if (!is_dir($d)) @mkdir($d, 0775, true);
    $real = realpath($d);
    if ($real === false) fail(500, "data_dir ($d) does not exist");
    return $real;
}

/**
 * Resolve a path relative to raw_root, refusing anything that climbs out of it.
 * Symbolic links you have made inside raw_root are followed (they are yours).
 */
function raw_path(string $rel, bool $mustBeDir = false): string {
    $root = raw_root();
    $rel = trim(str_replace('\\', '/', $rel), '/');
    if (str_contains($rel, "\0")) fail(400, 'Bad path');
    foreach (explode('/', $rel) as $seg) {
        if ($seg === '..' || $seg === '.') fail(400, 'That path is not inside the MAT folder');
    }
    $p = $rel === '' ? $root : $root . '/' . $rel;
    if (!file_exists($p)) fail(400, 'That file or folder no longer exists');
    if ($mustBeDir && !is_dir($p)) fail(400, 'Not a folder');
    return $p;
}

function rel_path(string $abs): string {
    $root = raw_root();
    if ($abs === $root) return '';
    return str_starts_with($abs, $root . '/') ? substr($abs, strlen($root) + 1) : $abs;
}

function safe_id(string $id): string {
    if (!preg_match('/^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$/', $id)) fail(400, 'Flight id: letters, digits, . _ - only (max 64), starting with a letter or digit');
    return $id;
}

function exporter_cmd(array $args): string {
    $parts = [escapeshellarg((string)admin_cfg('python')), '-u', escapeshellarg((string)admin_cfg('exporter'))];
    foreach ($args as $a) $parts[] = escapeshellarg((string)$a);
    return implode(' ', $parts);
}

function exec_available(): bool {
    $disabled = array_map('trim', explode(',', (string)ini_get('disable_functions')));
    return function_exists('shell_exec') && !in_array('shell_exec', $disabled, true);
}

function run_sync(array $args, int &$code = null): string {
    $out = [];
    exec(exporter_cmd($args) . ' 2>&1', $out, $code);
    return implode("\n", $out);
}

function read_json_file(string $f) {
    return is_file($f) ? json_decode((string)file_get_contents($f), true) : null;
}

function write_json_file(string $f, $data): void {
    $tmp = $f . '.tmp' . getmypid();
    file_put_contents($tmp, json_encode($data, JSON_PRETTY_PRINT | JSON_UNESCAPED_SLASHES | JSON_UNESCAPED_UNICODE));
    rename($tmp, $f);
}

/** Saved flight configs: id => config (with our extra 'source_dir'). */
function flight_configs(): array {
    $out = [];
    foreach (glob(work_dir('flights') . '/*.json') ?: [] as $f) {
        $c = read_json_file($f);
        if (is_array($c) && isset($c['id'])) $out[$c['id']] = $c;
    }
    return $out;
}

// ------------------------------------------------------------------ jobs
function job_file(string $id): string { return work_dir('jobs') . "/$id.json"; }

function job_status(array $j): array {
    $log = $j['log'] ?? '';
    $tail = '';
    if (is_file($log)) {
        $size = filesize($log);
        $fh = fopen($log, 'r');
        fseek($fh, max(0, $size - 16000));
        $tail = (string)stream_get_contents($fh);
        fclose($fh);
    }
    if (in_array($j['status'], ['running'], true)) {
        if (preg_match('/@exit (\d+)\s*$/', $tail, $m)) {
            $j['status'] = $m[1] === '0' ? 'done' : 'failed';
            $j['finished'] = $j['finished'] ?? gmdate('c');
            write_json_file(job_file($j['id']), $j);
        } elseif (!empty($j['pid']) && !file_exists('/proc/' . (int)$j['pid']) && !(function_exists('posix_kill') && posix_kill((int)$j['pid'], 0))) {
            $j['status'] = 'failed';
            $j['finished'] = gmdate('c');
            $j['error'] = 'The import stopped without finishing';
            write_json_file(job_file($j['id']), $j);
        }
    }
    $prog = null;
    if (preg_match_all('/@progress (\{.*\})/', $tail, $mm)) $prog = json_decode(end($mm[1]), true);
    $lines = array_values(array_filter(explode("\n", $tail), fn($l) => $l !== '' && $l[0] !== '@'));
    $j['progress'] = $prog;
    $j['log_tail'] = implode("\n", array_slice($lines, -14));
    unset($j['log'], $j['config_path'], $j['pid']);
    return $j;
}

/** Start queued jobs while fewer than max_jobs are running. Locked so two admins can't double-start. */
function pump_jobs(): void {
    if (!exec_available()) return;
    $lock = fopen(work_dir() . '/jobs.lock', 'c');
    if (!$lock || !flock($lock, LOCK_EX)) return;
    $jobs = [];
    foreach (glob(work_dir('jobs') . '/*.json') ?: [] as $f) { $j = read_json_file($f); if ($j) $jobs[] = $j; }
    usort($jobs, fn($a, $b) => strcmp($a['id'], $b['id']));
    $running = 0;
    foreach ($jobs as $j) if ($j['status'] === 'running' && job_status($j)['status'] === 'running') $running++;
    foreach ($jobs as $j) {
        if ($running >= (int)admin_cfg('max_jobs')) break;
        if ($j['status'] !== 'queued') continue;
        $args = [$j['config_path'], '--out', data_dir(), '--progress'];
        if (!empty($j['force'])) $args[] = '--force';
        $inner = exporter_cmd($args) . ' >> ' . escapeshellarg($j['log']) . ' 2>&1; echo "@exit $?" >> ' . escapeshellarg($j['log']);
        file_put_contents($j['log'], '$ ' . exporter_cmd($args) . "\n");
        $pid = (int)trim((string)shell_exec('nohup sh -c ' . escapeshellarg($inner) . ' > /dev/null 2>&1 & echo $!'));
        $j['status'] = 'running';
        $j['pid'] = $pid;
        $j['started'] = gmdate('c');
        write_json_file(job_file($j['id']), $j);
        $running++;
    }
    flock($lock, LOCK_UN);
    fclose($lock);
}

function queue_job(string $flightId, string $cfgPath, bool $force, string $user, string $kind = 'import'): array {
    $id = gmdate('Ymd-His') . '-' . bin2hex(random_bytes(3));
    $j = ['id' => $id, 'flight' => $flightId, 'kind' => $kind, 'status' => 'queued', 'force' => $force,
          'config_path' => $cfgPath, 'log' => work_dir('jobs') . "/$id.log", 'created' => gmdate('c'), 'user' => $user];
    write_json_file(job_file($id), $j);
    pump_jobs();
    return job_status(read_json_file(job_file($id)));
}

function rebuild_manifest(): void {
    $code = 0;
    $out = run_sync(['--manifest-only', '--out', data_dir()], $code);
    if ($code !== 0) fail(500, "Rebuilding the flight list failed:\n" . $out);
}

function rrmdir(string $dir): void {
    foreach (new RecursiveIteratorIterator(new RecursiveDirectoryIterator($dir, FilesystemIterator::SKIP_DOTS),
                                           RecursiveIteratorIterator::CHILD_FIRST) as $f) {
        $f->isDir() && !$f->isLink() ? rmdir($f->getPathname()) : unlink($f->getPathname());
    }
    rmdir($dir);
}

// ------------------------------------------------------------------ actions
function admin_action(string $action): void {
    $u = require_admin();
    if (str_starts_with($action, 'admin_project_')) {
        require __DIR__ . '/projects.php';
        project_action($action, $u);
    }
    switch ($action) {

    case 'admin_check':
        $checks = [];
        $add = function (string $name, bool $ok, string $detail, bool $optional = false) use (&$checks) {
            $checks[] = ['name' => $name, 'ok' => $ok, 'detail' => $detail, 'optional' => $optional];
        };
        $ex = exec_available();
        $add('PHP can run programs', $ex, $ex ? 'shell_exec is enabled' : 'shell_exec is disabled in php.ini (disable_functions)');
        $raw = admin_cfg('raw_root');
        $rawOk = $raw && is_dir($raw) && is_readable($raw);
        $add('MAT folder (raw_root)', (bool)$rawOk, $raw ? ($rawOk ? $raw : "$raw is missing or not readable") : 'not set in api/config.php');
        $wd = admin_cfg('work_dir');
        $wdOk = $wd && (is_dir($wd) || @mkdir($wd, 0775, true)) && is_writable($wd);
        $add('Work folder (work_dir)', (bool)$wdOk, $wd ? ($wdOk ? $wd : "$wd is missing or not writable by the web server") : 'not set in api/config.php');
        $dd = admin_cfg('data_dir');
        $ddOk = (is_dir($dd) || @mkdir($dd, 0775, true)) && is_writable($dd);
        $add('Web data folder (data_dir)', $ddOk, $ddOk ? (string)realpath($dd) : "$dd is not writable by the web server");
        $exp = admin_cfg('exporter');
        $add('Exporter script', is_file($exp), is_file($exp) ? (string)realpath($exp) : "$exp not found - set 'exporter' in config.php");
        if ($ex) {
            $py = trim((string)shell_exec(escapeshellarg((string)admin_cfg('python')) . ' -c ' . escapeshellarg(
                'import sys,json
r={"python":sys.version.split()[0]}
for m in ("numpy","scipy","netCDF4"):
    try:
        r[m]=__import__(m).__version__
    except Exception as e:
        r[m]=None
print(json.dumps(r))') . ' 2>&1'));
            $pj = json_decode($py, true);
            $add('Python', is_array($pj), is_array($pj) ? 'Python ' . $pj['python'] : 'could not run ' . admin_cfg('python') . ": $py");
            if (is_array($pj)) {
                $add('numpy and scipy', $pj['numpy'] && $pj['scipy'], $pj['numpy'] && $pj['scipy'] ? "numpy {$pj['numpy']}, scipy {$pj['scipy']}" : 'pip install numpy scipy');
                $add('netCDF4 (for .nc navigation files)', (bool)$pj['netCDF4'], $pj['netCDF4'] ? "netCDF4 {$pj['netCDF4']}" : 'optional: pip install netCDF4', true);
            }
        }
        out(['checks' => $checks]);

    case 'admin_scan':
        // Folders under raw_root holding MAT files and/or a pipeline particle_catalogue.csv.
        // Walks folders with glob (fast on big folders) and doesn't descend into the
        // PNG folders of a pipeline export.
        $root = raw_root();
        $cfgs = flight_configs();
        $bySource = [];
        foreach ($cfgs as $id => $c) $bySource[$c['source_dir'] ?? ''][] = $id;
        $dataDir = data_dir();
        $metasByConfig = [];
        foreach (glob("$dataDir/flights/*/meta.json") ?: [] as $mf) {
            $m = read_json_file($mf);
            if ($m && !empty($m['config_id'])) $metasByConfig[$m['config_id']][] = $m;
        }
        $skip = ['full_backgrounds.mat', 'timeseries.mat', 'timeseries_class.mat'];
        $list = [];
        $walk = function (string $dir, int $depth) use (&$walk, &$list, $root, $skip, $bySource, $cfgs, $dataDir, $metasByConfig) {
            $rel = rel_path($dir);
            $mats = [];
            $bytes = 0; $mtime = 0;
            foreach (glob($dir . '/*.mat') ?: [] as $f) {
                if (in_array(basename($f), $skip, true)) continue;
                $st = @stat($f);
                $mats[] = ['name' => basename($f), 'size' => $st['size'] ?? 0, 'mtime' => $st['mtime'] ?? 0];
                $bytes += $st['size'] ?? 0; $mtime = max($mtime, $st['mtime'] ?? 0);
            }
            $cat = is_file("$dir/particle_catalogue.csv");
            if ($mats || $cat) {
                $nav = [];
                foreach (array_merge(glob("$dir/*.csv") ?: [], glob("$dir/*.nc") ?: [], glob("$dir/*.nc4") ?: [], glob("$dir/*.txt") ?: []) as $f) {
                    if (in_array(basename($f), ['particle_catalogue.csv', 'flights.csv'], true)) continue;
                    $nav[] = ['name' => basename($f), 'size' => (int)@filesize($f), 'mtime' => (int)@filemtime($f)];
                }
                $flights = [];
                foreach ($bySource[$rel] ?? [] as $id) {
                    if (!empty($cfgs[$id]['catalogue_dir'])) {
                        $ms = $metasByConfig[$id] ?? [];
                        $flights[] = ['id' => $id, 'project' => true, 'exported' => (bool)$ms, 'n' => array_sum(array_map(fn($m) => $m['n'] ?? 0, $ms)),
                                      'flights' => count($ms), 'label' => $cfgs[$id]['campaign'] ?? $id];
                    } else {
                        $meta = read_json_file("$dataDir/flights/$id/meta.json");
                        $flights[] = ['id' => $id, 'exported' => (bool)$meta, 'exported_utc' => $meta['exported_utc'] ?? null,
                                      'n' => $meta['n'] ?? null, 'label' => $cfgs[$id]['label'] ?? $id];
                    }
                }
                $catInfo = null;
                if ($cat) {
                    $cf = "$dir/particle_catalogue.csv";
                    $catInfo = ['size' => (int)@filesize($cf), 'mtime' => (int)@filemtime($cf), 'flights_csv' => is_file("$dir/flights.csv")];
                }
                $list[] = ['dir' => $rel, 'mats' => $mats, 'nav' => $nav, 'bytes' => $bytes, 'mtime' => $mtime ?: ($catInfo['mtime'] ?? 0),
                           'catalogue' => $catInfo, 'flights' => $flights];
            }
            if ($cat || $depth >= 5) return;      // a pipeline export's subfolders only hold PNGs
            foreach (glob($dir . '/*', GLOB_ONLYDIR) ?: [] as $sub) {
                if (basename($sub)[0] === '.') continue;
                $walk($sub, $depth + 1);
            }
        };
        $walk($root, 0);
        usort($list, fn($a, $b) => strcmp($a['dir'], $b['dir']));
        out(['root' => $root, 'folders' => $list]);

    case 'admin_catsummary':
        $p = raw_path((string)($_GET['path'] ?? ''), true);
        $code = 0;
        $o = run_sync(['--catalogue-summary', $p], $code);
        $j = json_decode(trim((string)substr($o, (int)strrpos("\n" . $o, "\n{"))), true);
        if ($code !== 0 || !is_array($j)) fail(500, "Couldn't read the particle catalogue: " . $o);
        out(['summary' => $j]);

    case 'admin_save_project':
        // A pipeline export: {config, source_dir, mat_dir (relative to raw_root), groups: [keys], import, force}
        $b = require_post();
        $c = $b['config'] ?? null;
        if (!is_array($c)) fail(400, 'Missing project settings');
        $id = safe_id(trim((string)($c['id'] ?? '')));
        $dir = raw_path((string)($b['source_dir'] ?? ''), true);
        $rel = rel_path($dir);
        $existing = flight_configs()[$id] ?? null;
        if ($existing && ($existing['source_dir'] ?? null) !== $rel)
            fail(409, "The id $id is already used for " . ($existing['source_dir'] ?: '(top level)') . '. Choose another.');
        $matDir = trim((string)($b['mat_dir'] ?? ''));
        $matAbs = $matDir === '' ? $dir : raw_path($matDir, true);
        if (!glob($matAbs . '/*.mat')) fail(400, 'No MAT files in ' . ($matDir ?: $rel) . ' - say where the MAT files are');
        $groups = array_values(array_filter(array_map('strval', (array)($b['groups'] ?? [])), fn($g) => $g !== ''));
        if (!$groups) fail(400, 'Choose at least one flight');
        $c['id'] = $id;
        $c['catalogue_dir'] = $dir;
        $c['mat_dir'] = $matAbs;
        $c['groups'] = $groups;
        $c['source_dir'] = $rel;
        $c['id_prefix'] = $c['id_prefix'] ?? $id;
        $c['saved_by'] = $u['username'];
        $c['saved_utc'] = gmdate('c');
        $path = work_dir('flights') . "/$id.json";
        write_json_file($path, $c);
        $job = !empty($b['import']) ? queue_job($id, $path, !empty($b['force']), $u['username']) : null;
        out(['saved' => $id, 'job' => $job]);

    case 'admin_navcols':
        $p = raw_path((string)($_GET['path'] ?? ''));
        if (!is_file($p)) fail(400, 'Not a file');
        $code = 0;
        $o = run_sync(['--nav-columns', $p], $code);
        $j = json_decode(trim((string)substr($o, (int)strrpos("\n" . $o, "\n{"))), true);
        if ($code !== 0 || !is_array($j)) fail(500, "Couldn't read that file: " . $o);
        out(['nav' => $j]);

    case 'admin_flight_get':
        $id = safe_id((string)($_GET['id'] ?? ''));
        $c = read_json_file(work_dir('flights') . "/$id.json");
        if (!$c) fail(404, 'No saved settings for that flight');
        out(['config' => $c]);

    case 'admin_save_flight':
        // {config: {...flight.json fields...}, source_dir, mat_files: [names], nav_file: name|null, force, import: bool}
        $b = require_post();
        $c = $b['config'] ?? null;
        if (!is_array($c)) fail(400, 'Missing flight settings');
        $id = safe_id(trim((string)($c['id'] ?? '')));
        $dir = raw_path((string)($b['source_dir'] ?? ''), true);
        $rel = rel_path($dir);
        $existing = flight_configs()[$id] ?? null;
        if ($existing && ($existing['source_dir'] ?? null) !== $rel && empty($b['replace']))
            fail(409, "The id $id is already used for the folder " . ($existing['source_dir'] ?: '(top level)') . '. Choose another id.');
        $mats = [];
        foreach ((array)($b['mat_files'] ?? []) as $name) {
            $p = raw_path(($rel ? "$rel/" : '') . basename((string)$name));
            if (is_file($p) && str_ends_with(strtolower($p), '.mat')) $mats[] = $p;
        }
        if (!$mats) fail(400, 'Choose at least one MAT file');
        $c['id'] = $id;
        $c['mat_files'] = $mats;
        $c['source_dir'] = $rel;
        unset($c['nav']['file']);
        if (!empty($b['nav_file'])) {
            $np = raw_path(($rel ? "$rel/" : '') . basename((string)$b['nav_file']));
            $c['nav'] = array_merge((array)($c['nav'] ?? []), ['file' => $np]);
        } else {
            unset($c['nav']);
        }
        $c['saved_by'] = $u['username'];
        $c['saved_utc'] = gmdate('c');
        $path = work_dir('flights') . "/$id.json";
        write_json_file($path, $c);
        $job = !empty($b['import']) ? queue_job($id, $path, !empty($b['force']), $u['username']) : null;
        out(['saved' => $id, 'job' => $job]);

    case 'admin_import':
        // re-run the import for a saved flight
        $b = require_post();
        $id = safe_id((string)($b['id'] ?? ''));
        $path = work_dir('flights') . "/$id.json";
        if (!is_file($path)) fail(404, 'No saved settings for that flight');
        out(['job' => queue_job($id, $path, !empty($b['force']), $u['username'])]);

    case 'admin_jobs':
        pump_jobs();
        $jobs = [];
        foreach (glob(work_dir('jobs') . '/*.json') ?: [] as $f) { $j = read_json_file($f); if ($j) $jobs[] = $j; }
        usort($jobs, fn($a, $b) => strcmp($b['id'], $a['id']));
        $jobs = array_map('job_status', array_slice($jobs, 0, 25));
        out(['jobs' => $jobs]);

    case 'admin_job_cancel':
        $b = require_post();
        $f = job_file(preg_replace('/[^0-9A-Za-z-]/', '', (string)($b['id'] ?? '')));
        $j = read_json_file($f);
        if (!$j) fail(404, 'No such job');
        if ($j['status'] === 'running' && !empty($j['pid'])) {
            $pid = (int)$j['pid'];
            shell_exec("pkill -TERM -P $pid 2>/dev/null; kill -TERM $pid 2>/dev/null");
        }
        if (in_array($j['status'], ['running', 'queued'], true)) {
            $j['status'] = 'cancelled';
            $j['finished'] = gmdate('c');
            write_json_file($f, $j);
        }
        pump_jobs();
        out(['job' => job_status($j)]);

    case 'admin_flights':
        $dataDir = data_dir();
        $cfgs = flight_configs();
        $list = [];
        foreach (glob("$dataDir/flights/*/meta.json") ?: [] as $mf) {
            $m = read_json_file($mf);
            if (!$m) continue;
            $list[] = ['id' => $m['id'], 'label' => $m['label'] ?? $m['id'], 'campaign' => $m['campaign'] ?? null, 'n' => $m['n'] ?? 0, 'hidden' => !empty($m['hidden']),
                       'exported_utc' => $m['exported_utc'] ?? null, 'colormap' => $m['colormap'] ?? 'gray',
                       'config_id' => $m['config_id'] ?? (isset($cfgs[$m['id']]) ? $m['id'] : null),
                       'has_config' => isset($cfgs[$m['config_id'] ?? $m['id']]), 'source_dir' => $cfgs[$m['config_id'] ?? $m['id']]['source_dir'] ?? null];
        }
        usort($list, fn($a, $b) => strcmp(($a['campaign'] ?? '') . '/' . $a['id'], ($b['campaign'] ?? '') . '/' . $b['id']));
        out(['flights' => $list]);

    case 'admin_flight_hidden':
        $b = require_post();
        $id = safe_id((string)($b['id'] ?? ''));
        $mf = data_dir() . "/flights/$id/meta.json";
        $m = read_json_file($mf);
        if (!$m) fail(404, 'No exported flight with that id');
        $m['hidden'] = !empty($b['hidden']);
        write_json_file($mf, $m);
        rebuild_manifest();
        out(['id' => $id, 'hidden' => $m['hidden']]);

    case 'admin_flight_delete':
        // removes the exported images and data; keeps the saved settings and everyone's classifications
        $b = require_post();
        $id = safe_id((string)($b['id'] ?? ''));
        if (($b['confirm'] ?? '') !== $id) fail(400, 'Type the flight id to confirm');
        $dir = data_dir() . "/flights/$id";
        if (!is_dir($dir)) fail(404, 'No exported flight with that id');
        rrmdir($dir);
        if (!empty($b['forget'])) @unlink(work_dir('flights') . "/$id.json");
        rebuild_manifest();
        out(['deleted' => $id]);

    // ---------------------------------------------------------------- people
    case 'admin_users':
        $rows = db()->query('SELECT u.id, u.username, u.display_name, u.email, u.created_at, u.last_login, u.is_admin, u.disabled, u.must_change_pw,
                                    u.expertise, u.affiliation, u.is_external,
                                    (SELECT COUNT(*) FROM labels l WHERE l.user_id = u.id) AS n_labels
                             FROM users u ORDER BY u.display_name')->fetchAll();
        $cfgAdmins = cfg()['admins'] ?? [];
        foreach ($rows as &$r) {
            foreach (['id', 'is_admin', 'disabled', 'must_change_pw', 'n_labels', 'is_external'] as $k) $r[$k] = (int)$r[$k];
            $r['admin_in_config'] = in_array($r['username'], $cfgAdmins, true);
        }
        out(['users' => $rows, 'me' => $u['id'], 'expertise_levels' => expertise_levels()]);

    case 'admin_user_add':
        $b = require_post();
        $username = strtolower(trim((string)($b['username'] ?? '')));
        $display = trim((string)($b['display_name'] ?? ''));
        $email = trim((string)($b['email'] ?? '')) ?: null;
        if (!preg_match('/^[a-z0-9][a-z0-9_.-]{2,31}$/', $username)) fail(400, 'Username: 3-32 characters, lower-case letters, numbers, . _ -');
        if ($display === '' || mb_strlen($display) > 80) fail(400, 'Give the person’s name (up to 80 characters)');
        if ($email !== null && !filter_var($email, FILTER_VALIDATE_EMAIL)) fail(400, 'That email address does not look right');
        $st = db()->prepare('SELECT 1 FROM users WHERE username = ?');
        $st->execute([$username]);
        if ($st->fetch()) fail(409, "The username $username is taken");
        $temp = temp_password();
        $ex = trim((string)($b['expertise'] ?? ''));
        if ($ex !== '' && !in_array($ex, expertise_levels(), true)) $ex = '';
        db()->prepare('INSERT INTO users (username, display_name, email, pass_hash, created_at, is_admin, must_change_pw, disabled, expertise) VALUES (?,?,?,?,?,?,1,0,?)')
            ->execute([$username, $display, $email, password_hash($temp, PASSWORD_DEFAULT), now_sql(), !empty($b['is_admin']) ? 1 : 0, $ex ?: null]);
        out(['username' => $username, 'display_name' => $display, 'temp_password' => $temp]);

    case 'admin_user_reset':
        $b = require_post();
        $id = (int)($b['id'] ?? 0);
        $st = db()->prepare('SELECT username, display_name, is_external FROM users WHERE id = ?');
        $st->execute([$id]);
        $row = $st->fetch();
        if (!$row) fail(404, 'No such person');
        $temp = temp_password();
        // giving a password to someone who only had uploaded sessions turns them into a normal account
        db()->prepare('UPDATE users SET is_external = 0 WHERE id = ?')->execute([$id]);
        db()->prepare('UPDATE users SET pass_hash = ?, must_change_pw = 1 WHERE id = ?')->execute([password_hash($temp, PASSWORD_DEFAULT), $id]);
        out(['username' => $row['username'], 'display_name' => $row['display_name'], 'temp_password' => $temp]);

    case 'admin_user_update':
        $b = require_post();
        $id = (int)($b['id'] ?? 0);
        $st = db()->prepare('SELECT id, username, is_external FROM users WHERE id = ?');
        $st->execute([$id]);
        $row = $st->fetch();
        if (!$row) fail(404, 'No such person');
        if (isset($b['username'])) {
            // only for people added from a session file, before they get a login
            if (!(int)$row['is_external']) fail(400, 'Usernames can’t be changed once someone can sign in');
            $un = strtolower(trim((string)$b['username']));
            if (!preg_match('/^[a-z0-9][a-z0-9_.-]{2,31}$/', $un)) fail(400, 'Username: 3-32 characters, lower-case letters, numbers, . _ -');
            $chk = db()->prepare('SELECT 1 FROM users WHERE username = ? AND id <> ?');
            $chk->execute([$un, $id]);
            if ($chk->fetch()) fail(409, 'That username is taken');
            db()->prepare('UPDATE users SET username = ? WHERE id = ?')->execute([$un, $id]);
            $row['username'] = $un;
        }
        if ($id === $u['id'] && (isset($b['disabled']) && $b['disabled'] || isset($b['is_admin']) && !$b['is_admin']))
            fail(400, 'You can’t switch off your own account or your own administrator rights');
        if (isset($b['is_admin'])) db()->prepare('UPDATE users SET is_admin = ? WHERE id = ?')->execute([$b['is_admin'] ? 1 : 0, $id]);
        if (isset($b['disabled'])) db()->prepare('UPDATE users SET disabled = ? WHERE id = ?')->execute([$b['disabled'] ? 1 : 0, $id]);
        if (array_key_exists('expertise', $b)) {
            $ex = trim((string)$b['expertise']);
            if ($ex !== '' && !in_array($ex, expertise_levels(), true)) fail(400, 'Choose a level of expertise from the list');
            db()->prepare('UPDATE users SET expertise = ? WHERE id = ?')->execute([$ex ?: null, $id]);
        }
        if (array_key_exists('affiliation', $b))
            db()->prepare('UPDATE users SET affiliation = ? WHERE id = ?')->execute([mb_substr(trim((string)$b['affiliation']), 0, 120) ?: null, $id]);
        if (isset($b['display_name'])) {
            $dn = trim((string)$b['display_name']);
            if ($dn === '' || mb_strlen($dn) > 80) fail(400, 'Give a name (up to 80 characters)');
            db()->prepare('UPDATE users SET display_name = ? WHERE id = ?')->execute([$dn, $id]);
        }
        out(['id' => $id, 'username' => $row['username']]);

    // ---------------------------------------------------------------- classes
    case 'admin_scheme_save':
        $b = require_post();
        $sch = $b['scheme'] ?? null;
        if (!is_array($sch) || !is_array($sch['classes'] ?? null)) fail(400, 'Missing classes');
        $reserved = ['z', 'y', 'v', 'o', 'b', 'm', '?', ' '];
        $keys = []; $ids = [];
        $clean = ['title' => mb_substr(trim((string)($sch['title'] ?? 'Classes')), 0, 80),
                  'help' => mb_substr(trim((string)($sch['help'] ?? '')), 0, 600), 'classes' => [], 'flags' => []];
        foreach (['classes', 'flags'] as $kind) {
            foreach ((array)($sch[$kind] ?? []) as $c) {
                $name = trim((string)($c['name'] ?? ''));
                if ($name === '') fail(400, 'Every ' . ($kind === 'classes' ? 'class' : 'flag') . ' needs a name');
                $id = strtolower(trim((string)($c['id'] ?? '')));
                if ($id === '') $id = trim(preg_replace('/[^a-z0-9]+/', '_', strtolower($name)), '_');
                if (!preg_match('/^[a-z0-9_]{1,40}$/', $id)) fail(400, "Bad id for $name");
                if (isset($ids[$id])) fail(400, "Two entries share the id $id");
                $ids[$id] = 1;
                $key = mb_strtolower(mb_substr(trim((string)($c['key'] ?? '')), 0, 1));
                if ($key !== '') {
                    if (in_array($key, $reserved, true)) fail(400, "The key " . strtoupper($key) . " is used by the page itself (undo, view, outline...). Choose another for $name.");
                    if (isset($keys[$key])) fail(400, "The key " . strtoupper($key) . " is used twice ($name and {$keys[$key]})");
                    $keys[$key] = $name;
                }
                $item = ['id' => $id, 'name' => mb_substr($name, 0, 40), 'key' => $key, 'desc' => mb_substr(trim((string)($c['desc'] ?? '')), 0, 300)];
                if ($kind === 'classes') {
                    $col = (string)($c['color'] ?? '#888888');
                    if (!preg_match('/^#[0-9a-fA-F]{6}$/', $col)) $col = '#888888';
                    $item['color'] = $col;
                }
                $clean[$kind][] = $item;
            }
        }
        if (count($clean['classes']) < 2) fail(400, 'Keep at least two classes');
        $clean['version'] = 'custom-' . gmdate('YmdHi');
        $f = scheme_override_file();
        if (!$f) fail(500, "Set 'work_dir' in config.php so edited classes can be saved");
        work_dir();
        write_json_file($f, $clean);
        out(['scheme' => $clean]);

    case 'admin_scheme_reset':
        require_post();
        $f = scheme_override_file();
        if ($f && is_file($f)) @unlink($f);
        out([]);

    default:
        fail(404, 'Unknown admin action');
    }
}
