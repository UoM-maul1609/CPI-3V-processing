<?php
// Load a model's results (CSV from ml/habit_ml.py predict) into a classification set.
// Run by the machine-learning job after the model has finished; not from the web.
//   php web/api/tools/ml_load.php SET_ID results.csv
declare(strict_types=1);
if (PHP_SAPI !== 'cli') { http_response_code(404); exit; }
require __DIR__ . '/../lib.php';

[$sid, $csv] = [(int)($argv[1] ?? 0), (string)($argv[2] ?? '')];
if (!$sid || !is_file($csv)) { fwrite(STDERR, "usage: php ml_load.php SET_ID results.csv\n"); exit(2); }
ensure_schema();
$pdo = db();
$fh = fopen($csv, 'r');
$head = fgetcsv($fh);
$col = array_flip($head ?: []);
foreach (['pid', 'flight_id', 'label', 'prob'] as $c) if (!isset($col[$c])) { fwrite(STDERR, "missing column $c\n"); exit(2); }

$n = 0;
$rows = [];
$flush = function () use (&$rows, $pdo, $sid) {
    if (!$rows) return;
    $sql = (is_sqlite() ? 'INSERT OR REPLACE' : 'REPLACE') . ' INTO ml_predictions (set_id, pid, flight_id, label, prob, label2, prob2, riming, riming_prob, flags) VALUES '
         . implode(',', array_fill(0, count($rows), '(?,?,?,?,?,?,?,?,?,?)'));
    $args = [];
    foreach ($rows as $r) array_push($args, $sid, ...$r);
    $pdo->prepare($sql)->execute($args);
    $rows = [];
};
$pdo->beginTransaction();
try {
    $pdo->prepare('DELETE FROM ml_predictions WHERE set_id = ?')->execute([$sid]);
    while (($r = fgetcsv($fh)) !== false) {
        if (count($r) < 4 || !preg_match(PID_RE, $r[$col['pid']])) continue;
        $l2 = isset($col['label2']) && ($r[$col['label2']] ?? '') !== '' ? $r[$col['label2']] : null;
        $p2 = isset($col['prob2']) && ($r[$col['prob2']] ?? '') !== '' ? (float)$r[$col['prob2']] : null;
        $opt = fn($c) => isset($col[$c]) && ($r[$col[$c]] ?? '') !== '' ? $r[$col[$c]] : null;
        $rp = $opt('riming_prob');
        $rows[] = [$r[$col['pid']], $r[$col['flight_id']], $r[$col['label']], (float)$r[$col['prob']], $l2, $p2,
                   $opt('riming'), $rp === null ? null : (float)$rp, $opt('flags') === null ? null : mb_substr(str_replace(';', ',', $opt('flags')), 0, 200)];
        $n++;
        if (count($rows) >= (is_sqlite() ? 90 : 400)) $flush();
        if ($n % 100000 === 0) echo "loaded $n\n";
    }
    $flush();
    $pdo->prepare("UPDATE ml_sets SET n = ?, status = 'ready' WHERE id = ?")->execute([$n, $sid]);
    $pdo->commit();
} catch (Throwable $e) {
    if ($pdo->inTransaction()) $pdo->rollBack();
    $pdo->prepare("UPDATE ml_sets SET status = 'failed' WHERE id = ?")->execute([$sid]);
    fwrite(STDERR, 'Loading the results failed: ' . $e->getMessage() . "\n");
    exit(1);
}
echo "Saved $n results in classification set $sid\n";
