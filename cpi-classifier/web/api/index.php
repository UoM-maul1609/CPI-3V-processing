<?php
// CPI classifier API. All requests: api/?a=<action>
// The browser does the heavy lifting (filtering, sorting, sampling); the
// server only handles accounts and reading/writing classifications.
declare(strict_types=1);
require __DIR__ . '/lib.php';

$action = $_GET['a'] ?? '';

switch ($action) {

// ---------------------------------------------------------------- session
case 'me':
    $c = cfg();
    if (is_demo()) out(['demo' => true, 'user' => null, 'setup' => false, 'allow_registration' => false, 'needs_code' => false, 'expertise_levels' => expertise_levels()]);
    ensure_schema();
    $u = current_user();
    out(['user' => $u, 'demo' => false, 'setup' => user_count() === 0, 'expertise_levels' => expertise_levels(),
         'allow_registration' => (bool)($c['allow_registration'] ?? false),
         'needs_code' => ($c['registration_code'] ?? '') !== '']);

case 'scheme':
    header('Cache-Control: no-cache');
    out(['scheme' => scheme()]);

case 'register':
    // Open registration (if allowed in config), or the very first account,
    // which becomes the administrator.
    if (is_demo()) fail(403, 'This is a demonstration site: accounts are switched off');
    $b = require_post();
    $c = cfg();
    ensure_schema();
    $first = user_count() === 0;
    if (!$first) {
        if (empty($c['allow_registration'])) fail(403, 'Ask the administrator to add you: they’ll give you a username and a temporary password');
        if (($c['registration_code'] ?? '') !== '' && !hash_equals((string)$c['registration_code'], (string)($b['code'] ?? '')))
            fail(403, 'That registration code is not right');
    }
    $username = strtolower(trim((string)($b['username'] ?? '')));
    $display = trim((string)($b['display_name'] ?? ''));
    $pass = (string)($b['password'] ?? '');
    $email = trim((string)($b['email'] ?? '')) ?: null;
    if (!preg_match('/^[a-z0-9][a-z0-9_.-]{2,31}$/', $username)) fail(400, 'Username: 3-32 characters, letters, numbers, . _ -');
    if ($display === '' || mb_strlen($display) > 80) fail(400, 'Please give your name (up to 80 characters)');
    if ($e = password_problem($pass, $username)) fail(400, $e);
    $admins = $c['admins'] ?? [];
    if ($first && $admins && !in_array($username, $admins, true))
        fail(403, 'The first account must use one of the administrator usernames in api/config.php (' . implode(', ', $admins) . ')');
    if ($email !== null && !filter_var($email, FILTER_VALIDATE_EMAIL)) fail(400, 'That email address does not look right');
    $st = db()->prepare('SELECT 1 FROM users WHERE username = ?');
    $st->execute([$username]);
    if ($st->fetch()) fail(409, 'That username is taken');
    db()->prepare('INSERT INTO users (username, display_name, email, pass_hash, created_at, last_login, is_admin, must_change_pw) VALUES (?,?,?,?,?,?,?,0)')
        ->execute([$username, $display, $email, password_hash($pass, PASSWORD_DEFAULT), now_sql(), now_sql(), $first ? 1 : 0]);
    start_session();
    session_regenerate_id(true);
    $_SESSION['uid'] = (int)db()->lastInsertId();
    out(['user' => current_user()]);

case 'login':
    if (is_demo()) fail(403, 'This is a demonstration site: signing in is switched off');
    $b = require_post();
    ensure_schema();
    $username = strtolower(trim((string)($b['username'] ?? '')));
    $st = db()->prepare('SELECT id, pass_hash, disabled FROM users WHERE username = ?');
    $st->execute([$username]);
    $row = $st->fetch();
    if (!$row || !password_verify((string)($b['password'] ?? ''), $row['pass_hash'])) {
        usleep(400000);
        fail(401, 'Wrong username or password');
    }
    if ((int)$row['disabled']) fail(403, 'This account has been switched off. Ask the administrator.');
    if (password_needs_rehash($row['pass_hash'], PASSWORD_DEFAULT)) {
        db()->prepare('UPDATE users SET pass_hash = ? WHERE id = ?')
            ->execute([password_hash((string)$b['password'], PASSWORD_DEFAULT), $row['id']]);
    }
    db()->prepare('UPDATE users SET last_login = ? WHERE id = ?')->execute([now_sql(), $row['id']]);
    start_session();
    session_regenerate_id(true);
    $_SESSION['uid'] = (int)$row['id'];
    out(['user' => current_user()]);

case 'change_password':
    $u = require_user(true);
    $b = require_post();
    $st = db()->prepare('SELECT pass_hash FROM users WHERE id = ?');
    $st->execute([$u['id']]);
    if (!password_verify((string)($b['current'] ?? ''), (string)$st->fetchColumn())) fail(401, 'Your current password isn’t right');
    $new = (string)($b['new'] ?? '');
    if ($e = password_problem($new, $u['username'])) fail(400, $e);
    if ($new === (string)$b['current']) fail(400, 'Choose a password different from the temporary one');
    db()->prepare('UPDATE users SET pass_hash = ?, must_change_pw = 0 WHERE id = ?')->execute([password_hash($new, PASSWORD_DEFAULT), $u['id']]);
    session_regenerate_id(true);
    out(['user' => current_user(true)]);

case 'profile_update':
    // your own name, level of expertise and affiliation
    $u = require_user(true);
    $b = require_post();
    if (isset($b['display_name'])) {
        $dn = trim((string)$b['display_name']);
        if ($dn === '' || mb_strlen($dn) > 80) fail(400, 'Give your name (up to 80 characters)');
        db()->prepare('UPDATE users SET display_name = ? WHERE id = ?')->execute([$dn, $u['id']]);
    }
    if (array_key_exists('expertise', $b)) {
        $ex = trim((string)$b['expertise']);
        if ($ex !== '' && !in_array($ex, expertise_levels(), true)) fail(400, 'Choose a level of expertise from the list');
        db()->prepare('UPDATE users SET expertise = ? WHERE id = ?')->execute([$ex ?: null, $u['id']]);
    }
    if (array_key_exists('affiliation', $b)) {
        db()->prepare('UPDATE users SET affiliation = ? WHERE id = ?')->execute([mb_substr(trim((string)$b['affiliation']), 0, 120) ?: null, $u['id']]);
    }
    out(['user' => current_user(true)]);

case 'prefs_save':
    // display settings (colour map, outline, scale...) kept with the account
    $u = require_user(true);
    $b = require_post();
    $prefs = $b['prefs'] ?? null;
    if (!is_array($prefs)) fail(400, 'Missing settings');
    $json = json_encode($prefs, JSON_UNESCAPED_UNICODE);
    if (strlen($json) > 20000) fail(400, 'Settings too large');
    db()->prepare('UPDATE users SET prefs = ? WHERE id = ?')->execute([$json, $u['id']]);
    out([]);

case 'logout':
    require_post();
    start_session();
    $_SESSION = [];
    session_destroy();
    out([]);

// ---------------------------------------------------------- classifications
case 'labels':
    if (is_demo()) out(['mine' => (object)[], 'counts' => (object)[]]);
    // ?flights=A,B  -> my labels and everyone's counts for those flights
    $flights = array_values(array_filter(explode(',', (string)($_GET['flights'] ?? '')),
        fn($f) => preg_match('/^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$/', $f)));
    if (!$flights) out(['mine' => (object)[], 'counts' => (object)[]]);
    $in = implode(',', array_fill(0, count($flights), '?'));
    $u = current_user();
    $mine = [];
    if ($u) {
        $st = db()->prepare("SELECT pid, label, flags, made_of FROM labels WHERE user_id = ? AND flight_id IN ($in)");
        $st->execute(array_merge([$u['id']], $flights));
        foreach ($st as $r) $mine[$r['pid']] = $r['made_of'] !== null ? [$r['label'], $r['flags'], $r['made_of']] : [$r['label'], $r['flags']];
    }
    // counts per particle and per label (all users) -> {pid: {label: n}}
    $st = db()->prepare("SELECT pid, label, COUNT(*) AS n FROM labels WHERE flight_id IN ($in) GROUP BY pid, label");
    $st->execute($flights);
    $counts = [];
    foreach ($st as $r) $counts[$r['pid']][$r['label']] = (int)$r['n'];
    out(['mine' => (object)$mine, 'counts' => (object)$counts]);

case 'save':
    // {items: [{pid, label|null, flags: [..], made_of?, note?}], scheme_version}
    $u = require_user();
    $b = require_post();
    $items = $b['items'] ?? [];
    if (!is_array($items) || count($items) > 1000) fail(400, 'Send between 1 and 1000 items');
    $sch = scheme();
    $okLabels = array_column($sch['classes'] ?? [], 'id');
    $okFlags = array_column($sch['flags'] ?? [], 'id');
    $forMap = flag_for_map($sch);
    $ofMap = made_of_map($sch);
    $ver = substr((string)($b['scheme_version'] ?? ($sch['version'] ?? '')), 0, 32);
    $now = now_sql();
    $pdo = db();
    $upsert = is_sqlite()
        ? 'INSERT INTO labels (user_id,pid,flight_id,label,flags,note,scheme_version,created_at,updated_at,made_of) VALUES (?,?,?,?,?,?,?,?,?,?)
           ON CONFLICT(user_id,pid) DO UPDATE SET label=excluded.label, flags=excluded.flags, note=excluded.note,
           scheme_version=excluded.scheme_version, updated_at=excluded.updated_at, session_id=NULL, made_of=excluded.made_of'
        : 'INSERT INTO labels (user_id,pid,flight_id,label,flags,note,scheme_version,created_at,updated_at,made_of) VALUES (?,?,?,?,?,?,?,?,?,?)
           ON DUPLICATE KEY UPDATE label=VALUES(label), flags=VALUES(flags), note=VALUES(note),
           scheme_version=VALUES(scheme_version), updated_at=VALUES(updated_at), session_id=NULL, made_of=VALUES(made_of)';
    $ins = $pdo->prepare($upsert);
    $del = $pdo->prepare('DELETE FROM labels WHERE user_id = ? AND pid = ?');
    $logq = $pdo->prepare('INSERT INTO label_log (user_id,pid,label,flags,created_at,made_of) VALUES (?,?,?,?,?,?)');
    $saved = 0;
    $pdo->beginTransaction();
    try {
        foreach ($items as $it) {
            $pid = (string)($it['pid'] ?? '');
            if (!preg_match(PID_RE, $pid, $m)) continue;
            $label = $it['label'] ?? null;
            if ($label === null || $label === '') {
                $del->execute([$u['id'], $pid]);
                $logq->execute([$u['id'], $pid, null, null, $now, null]);
                $saved++;
                continue;
            }
            if ($okLabels && !in_array($label, $okLabels, true)) continue;
            $flags = array_values(array_filter((array)($it['flags'] ?? []),
                fn($f) => is_string($f) && (!$okFlags || in_array($f, $okFlags, true)) && flag_applies($forMap, $f, $label)));
            $flagStr = implode(',', $flags);
            $note = isset($it['note']) ? mb_substr((string)$it['note'], 0, 500) : null;
            $mo = clean_made_of($ofMap, $label, $it['made_of'] ?? null);
            $ins->execute([$u['id'], $pid, $m[1], $label, $flagStr, $note, $ver, $now, $now, $mo]);
            $logq->execute([$u['id'], $pid, $label, $flagStr, $now, $mo]);
            $saved++;
        }
        $pdo->commit();
    } catch (Throwable $e) {
        $pdo->rollBack();
        error_log('cpi-classifier save: ' . $e->getMessage());
        fail(500, 'Could not save');
    }
    out(['saved' => $saved]);

case 'examples':
    // Example particles for the Guide: for each class (and each "made of"), particles that the
    // people who classified them agree on, from all classifications in the database.
    // ?flights=A,B (optional: only these flights)  &per=8  &seed=N (a different random choice)
    // &mine=1: the signed-in person's own classifications instead, grouped the same way.
    // Each example also carries "me": [label, flags, made_of] when the signed-in person has classified it.
    if (is_demo()) out(['examples' => (object)[]]);
    ensure_schema();
    $per = max(1, min(60, (int)($_GET['per'] ?? 8)));
    $seed = (string)($_GET['seed'] ?? '');
    $where = ''; $args = [];
    if (isset($_GET['flights'])) {
        $fl = array_values(array_filter(explode(',', (string)$_GET['flights']), fn($f) => preg_match('/^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$/', $f)));
        if (!$fl) out(['examples' => (object)[]]);
        $where = 'WHERE flight_id IN (' . implode(',', array_fill(0, count($fl), '?')) . ')';
        $args = $fl;
    }
    $me = current_user();
    if (!empty($_GET['mine'])) {
        if (!$me) out(['examples' => (object)[]]);
        $w2 = $where ? "$where AND user_id = ?" : 'WHERE user_id = ?';
        $st = db()->prepare("SELECT pid, label, flags, made_of FROM labels $w2");
        $st->execute(array_merge($args, [$me['id']]));
        $cand = [];
        foreach ($st as $r) {
            $e = [crc32($r['pid'] . $seed), $r['pid'], $r['made_of'], 1, 1, [$r['label'], (string)$r['flags'], $r['made_of']]];
            $cand[$r['label']][] = $e;
            if ($r['made_of'] !== null) $cand[$r['label'] . ':' . $r['made_of']][] = $e;
        }
        $outEx = [];
        foreach ($cand as $g => $list) {
            usort($list, fn($a, $b) => $a[0] <=> $b[0]);
            $outEx[$g] = array_map(fn($x) => ['pid' => $x[1], 'made_of' => $x[2], 'n' => 1, 't' => 1, 'me' => $x[5]], array_slice($list, 0, $per));
            $outEx[$g . '#count'] = count($list);
        }
        out(['examples' => (object)$outEx]);
    }
    $tot = [];
    $st = db()->prepare("SELECT pid, COUNT(*) AS t FROM labels $where GROUP BY pid");
    $st->execute($args);
    foreach ($st as $r) $tot[$r['pid']] = (int)$r['t'];
    $st = db()->prepare("SELECT pid, label, made_of, COUNT(*) AS n FROM labels $where GROUP BY pid, label, made_of");
    $st->execute($args);
    $byPidLabel = [];   // pid|label -> [n, [made_of => n]]
    foreach ($st as $r) {
        $k = $r['pid'] . "\t" . $r['label'];
        $byPidLabel[$k][0] = ($byPidLabel[$k][0] ?? 0) + (int)$r['n'];
        if ($r['made_of'] !== null) $byPidLabel[$k][1][$r['made_of']] = ($byPidLabel[$k][1][$r['made_of']] ?? 0) + (int)$r['n'];
    }
    $cand = [];   // group ("label" or "label:made_of") -> [[score, pid, made_of, n, t]]
    foreach ($byPidLabel as $k => [$n, $ofs]) {
        [$pid, $label] = explode("\t", $k);
        $t = $tot[$pid];
        if ($n / $t < 0.66) continue;                                   // most people agree
        $score = $n * 10 - ($t - $n) * 15 + (crc32($pid . $seed) % 1000) / 1000;   // agreement first, then random
        $mo = null;
        if (!empty($ofs)) { arsort($ofs); $mo = array_key_first($ofs); if ($ofs[$mo] / $n < 0.5) $mo = null; }
        $cand[$label][] = [$score, $pid, $mo, $n, $t];
        if ($mo !== null) $cand["$label:$mo"][] = [$score, $pid, $mo, $n, $t];
    }
    $outEx = [];
    foreach ($cand as $g => $list) {
        usort($list, fn($a, $b) => $b[0] <=> $a[0]);
        $outEx[$g] = array_map(fn($x) => ['pid' => $x[1], 'made_of' => $x[2], 'n' => $x[3], 't' => $x[4]], array_slice($list, 0, $per));
        $outEx[$g . '#count'] = count($list);
    }
    if ($me && $outEx) {
        // what the signed-in person said about each example
        $pids = [];
        foreach ($outEx as $g => $list) if (is_array($list)) foreach ($list as $e) $pids[$e['pid']] = 1;
        $mine = [];
        foreach (array_chunk(array_keys($pids), 500) as $chunk) {
            $st = db()->prepare('SELECT pid, label, flags, made_of FROM labels WHERE user_id = ? AND pid IN (' . implode(',', array_fill(0, count($chunk), '?')) . ')');
            $st->execute(array_merge([$me['id']], $chunk));
            foreach ($st as $r) $mine[$r['pid']] = [$r['label'], (string)$r['flags'], $r['made_of']];
        }
        foreach ($outEx as $g => &$list) if (is_array($list)) foreach ($list as &$e) if (isset($mine[$e['pid']])) $e['me'] = $mine[$e['pid']];
        unset($list, $e);
    }
    out(['examples' => (object)$outEx]);

case 'session_upload':
    // Classifications from a session file (practice mode download), saved as one person's work.
    // {meta: {name, expertise, affiliation, notes, file_name, scheme_version, started_utc},
    //  target: 'me' | 'new' | <user id>, clash: 'newer' | 'file' | 'keep', items: [{pid, label, flags, t}]}
    // Every classification in the file goes into the person's history (label_log) with the date it was
    // made. Where they had already classified a particle, 'newer' (the default) makes whichever
    // opinion is more recent their current one; the other stays in the history.
    $u = require_user();
    $b = require_post();
    $meta = (array)($b['meta'] ?? []);
    $items = $b['items'] ?? [];
    if (!is_array($items) || !$items) fail(400, 'The file has no classifications in it');
    if (count($items) > 100000) fail(400, 'That file is too big (more than 100,000 classifications)');
    $clip = fn($k, $n) => ($v = trim((string)($meta[$k] ?? ''))) === '' ? null : mb_substr($v, 0, $n);
    $name = $clip('name', 80); $exp = $clip('expertise', 60); $aff = $clip('affiliation', 120);
    $target = (string)($b['target'] ?? 'me');
    $pdo = db();
    $pdo->beginTransaction();
    try {
        if ($target === 'me') {
            $cid = $u['id'];
        } else {
            if (!$u['is_admin']) fail(403, 'Only administrators can add classifications on someone else’s behalf');
            if ($target === 'new') {
                if (!$name) fail(400, 'Give the name of the person who made these classifications');
                $slug = trim(preg_replace('/[^a-z0-9]+/', '-', strtolower(iconv('UTF-8', 'ASCII//TRANSLIT', $name) ?: $name)), '-') ?: 'person';
                $username = 'guest-' . substr($slug, 0, 20);
                $chk = $pdo->prepare('SELECT 1 FROM users WHERE username = ?');
                for ($i = 2, $base = $username; ; $i++) { $chk->execute([$username]); if (!$chk->fetch()) break; $username = "$base-$i"; }
                $pdo->prepare('INSERT INTO users (username, display_name, pass_hash, created_at, is_admin, must_change_pw, disabled, expertise, affiliation, is_external) VALUES (?,?,?,?,0,0,0,?,?,1)')
                    ->execute([$username, $name, '!', now_sql(), $exp, $aff]);
                $cid = (int)$pdo->lastInsertId();
            } else {
                $cid = (int)$target;
                $st = $pdo->prepare('SELECT id FROM users WHERE id = ?');
                $st->execute([$cid]);
                if (!$st->fetch()) fail(404, 'No such person');
            }
        }
        $sch = scheme();
        $okLabels = array_column($sch['classes'] ?? [], 'id');
        $okFlags = array_column($sch['flags'] ?? [], 'id');
        $forMap = flag_for_map($sch);
        $pdo->prepare('INSERT INTO sessions (contributor_id, uploaded_by, name, expertise, affiliation, notes, file_name, scheme_version, started_utc, n_labels, created_at) VALUES (?,?,?,?,?,?,?,?,?,0,?)')
            ->execute([$cid, $u['id'], $name, $exp, $aff, $clip('notes', 1000), $clip('file_name', 200), $clip('scheme_version', 32), $clip('started_utc', 32), now_sql()]);
        $sid = (int)$pdo->lastInsertId();
        $clash = (string)($b['clash'] ?? (isset($b['replace']) ? (!empty($b['replace']) ? 'file' : 'keep') : 'newer'));
        if (!in_array($clash, ['newer', 'file', 'keep'], true)) $clash = 'newer';
        $upsert = is_sqlite()
            ? 'INSERT INTO labels (user_id,pid,flight_id,label,flags,note,scheme_version,created_at,updated_at,session_id,made_of) VALUES (?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(user_id,pid) DO UPDATE SET label=excluded.label, flags=excluded.flags, scheme_version=excluded.scheme_version,
               updated_at=excluded.updated_at, session_id=excluded.session_id, made_of=excluded.made_of'
            : 'INSERT INTO labels (user_id,pid,flight_id,label,flags,note,scheme_version,created_at,updated_at,session_id,made_of) VALUES (?,?,?,?,?,?,?,?,?,?,?)
               ON DUPLICATE KEY UPDATE label=VALUES(label), flags=VALUES(flags), scheme_version=VALUES(scheme_version),
               updated_at=VALUES(updated_at), session_id=VALUES(session_id), made_of=VALUES(made_of)';
        $ins = $pdo->prepare($upsert);
        $has = $pdo->prepare('SELECT label, flags, made_of, updated_at FROM labels WHERE user_id = ? AND pid = ?');
        $logged = $pdo->prepare('SELECT 1 FROM label_log WHERE user_id = ? AND pid = ? LIMIT 1');
        $logq = $pdo->prepare('INSERT INTO label_log (user_id,pid,label,flags,created_at,made_of) VALUES (?,?,?,?,?,?)');
        $ofMap = made_of_map($sch);
        $saved = $bad = $kept = $replaced = $same = 0;
        $unknown = [];
        $ver = $clip('scheme_version', 32) ?? ($sch['version'] ?? null);
        foreach ($items as $it) {
            $pid = (string)($it['pid'] ?? '');
            if (!preg_match(PID_RE, $pid, $m)) { $bad++; continue; }
            $label = (string)($it['label'] ?? '');
            if ($okLabels && !in_array($label, $okLabels, true)) { $unknown[$label] = ($unknown[$label] ?? 0) + 1; continue; }
            $flags = implode(',', array_values(array_filter((array)($it['flags'] ?? []), fn($f) => is_string($f) && (!$okFlags || in_array($f, $okFlags, true)) && flag_applies($forMap, $f, (string)$label))));
            $when = isset($it['t']) && is_numeric($it['t']) ? gmdate('Y-m-d H:i:s', (int)$it['t']) : now_sql();
            $mo = clean_made_of($ofMap, $label, $it['made_of'] ?? null);
            $has->execute([$cid, $pid]);
            $cur = $has->fetch();
            $has->closeCursor();
            if ($cur) {
                if ($cur['label'] === $label && (string)$cur['flags'] === $flags && ($cur['made_of'] ?? null) === $mo) { $same++; continue; }
                // make sure the opinion already held is in the history before anything changes
                $logged->execute([$cid, $pid]);
                if (!$logged->fetch()) $logq->execute([$cid, $pid, $cur['label'], $cur['flags'], $cur['updated_at'], $cur['made_of']]);
                $logged->closeCursor();
                $logq->execute([$cid, $pid, $label, $flags, $when, $mo]);      // the file's opinion, with its own date
                $fileWins = $clash === 'file' || ($clash === 'newer' && strcmp($when, (string)$cur['updated_at']) > 0);
                if (!$fileWins) { $kept++; continue; }
                $ins->execute([$cid, $pid, $m[1], $label, $flags, null, $ver, $when, $when, $sid, $mo]);
                $replaced++;
                continue;
            }
            $ins->execute([$cid, $pid, $m[1], $label, $flags, null, $ver, $when, $when, $sid, $mo]);
            $logq->execute([$cid, $pid, $label, $flags, $when, $mo]);
            $saved++;
        }
        $pdo->prepare('UPDATE sessions SET n_labels = ? WHERE id = ?')->execute([$saved + $replaced, $sid]);
        $pdo->commit();
    } catch (Throwable $e) {
        if ($pdo->inTransaction()) $pdo->rollBack();
        error_log('cpi-classifier session_upload: ' . $e->getMessage());
        fail(500, 'Could not save the session');
    }
    $st = db()->prepare('SELECT display_name FROM users WHERE id = ?');
    $st->execute([$cid]);
    out(['session_id' => $sid, 'saved' => $saved, 'replaced' => $replaced, 'same' => $same, 'kept_existing' => $kept, 'bad' => $bad, 'unknown' => $unknown, 'contributor' => $st->fetchColumn()]);

case 'sessions':
    $u = require_user();
    $sql = 'SELECT s.*, c.display_name AS contributor, c.is_external, up.display_name AS uploader,
                   (SELECT COUNT(*) FROM labels l WHERE l.session_id = s.id) AS n_current
            FROM sessions s JOIN users c ON c.id = s.contributor_id JOIN users up ON up.id = s.uploaded_by';
    $args = [];
    if (!$u['is_admin']) { $sql .= ' WHERE s.uploaded_by = ? OR s.contributor_id = ?'; $args = [$u['id'], $u['id']]; }
    $st = db()->prepare($sql . ' ORDER BY s.created_at DESC');
    $st->execute($args);
    out(['sessions' => $st->fetchAll()]);

case 'session_delete':
    // removes the classifications that came from this session (not later edits)
    $u = require_user();
    $b = require_post();
    $id = (int)($b['id'] ?? 0);
    $st = db()->prepare('SELECT * FROM sessions WHERE id = ?');
    $st->execute([$id]);
    $sess = $st->fetch();
    if (!$sess) fail(404, 'No such session');
    if (!$u['is_admin'] && (int)$sess['uploaded_by'] !== $u['id']) fail(403, 'Only the person who uploaded it, or an administrator, can remove it');
    $pdo = db();
    $pdo->beginTransaction();
    $pdo->prepare('INSERT INTO label_log (user_id,pid,label,flags,created_at) SELECT user_id, pid, NULL, NULL, ? FROM labels WHERE session_id = ?')->execute([now_sql(), $id]);
    $del = $pdo->prepare('DELETE FROM labels WHERE session_id = ?');
    $del->execute([$id]);
    $removed = $del->rowCount();
    $pdo->prepare('DELETE FROM sessions WHERE id = ?')->execute([$id]);
    // a contributor without a login who has nothing left is removed too
    $c = (int)$sess['contributor_id'];
    $left = $pdo->prepare('SELECT (SELECT COUNT(*) FROM labels WHERE user_id = ?) + (SELECT COUNT(*) FROM sessions WHERE contributor_id = ?)');
    $left->execute([$c, $c]);
    if ((int)$left->fetchColumn() === 0) $pdo->prepare('DELETE FROM users WHERE id = ? AND is_external = 1')->execute([$c]);
    $pdo->commit();
    out(['removed' => $removed]);

case 'stats':
    if (is_demo()) out(['demo' => true, 'by_label' => [], 'by_flight' => [], 'people' => [], 'mine' => [], 'my_days' => [], 'agreement' => ['multi' => 0, 'unanimous' => 0]]);
    $u = current_user();
    $byLabel = db()->query('SELECT label, COUNT(*) AS n FROM labels GROUP BY label')->fetchAll();
    $byFlight = db()->query('SELECT flight_id, COUNT(*) AS n, COUNT(DISTINCT pid) AS particles FROM labels GROUP BY flight_id')->fetchAll();
    $people = db()->query('SELECT u.display_name AS name, u.expertise, u.is_external, COUNT(l.id) AS n, MAX(l.updated_at) AS last
                           FROM users u JOIN labels l ON l.user_id = u.id GROUP BY u.id, u.display_name, u.expertise, u.is_external ORDER BY n DESC LIMIT 50')->fetchAll();
    $mine = [];
    $recent = [];
    if ($u) {
        $st = db()->prepare('SELECT label, COUNT(*) AS n FROM labels WHERE user_id = ? GROUP BY label');
        $st->execute([$u['id']]);
        $mine = $st->fetchAll();
        $st = db()->prepare("SELECT substr(updated_at,1,10) AS day, COUNT(*) AS n FROM labels WHERE user_id = ? GROUP BY substr(updated_at,1,10) ORDER BY day DESC LIMIT 30");
        $st->execute([$u['id']]);
        $recent = $st->fetchAll();
    }
    // agreement: particles with 2+ classifications, and how many all agree
    $agree = db()->query('SELECT COUNT(*) AS multi, SUM(CASE WHEN nl = 1 THEN 1 ELSE 0 END) AS unanimous
                          FROM (SELECT pid, COUNT(DISTINCT label) AS nl, COUNT(*) AS n FROM labels GROUP BY pid HAVING COUNT(*) > 1) x')->fetch();
    out(['by_label' => $byLabel, 'by_flight' => $byFlight, 'people' => $people,
         'mine' => $mine, 'my_days' => $recent, 'agreement' => $agree]);

case 'export':
    // Admins: everyone's labels. Others: their own.
    // &history=1: every opinion with its date (label_log), so changes of mind can be followed.
    $u = require_user();
    $fmt = $_GET['format'] ?? 'csv';
    if (($_GET['history'] ?? '') === '1') {
        $sql = 'SELECT g.pid, g.label, g.made_of, g.flags, g.created_at, u.username, u.display_name, u.expertise
                FROM label_log g JOIN users u ON u.id = g.user_id';
        $args = [];
        if (!$u['is_admin'] || ($_GET['mine'] ?? '') === '1') { $sql .= ' WHERE g.user_id = ?'; $args[] = $u['id']; }
        $sql .= ' ORDER BY g.pid, u.username, g.created_at, g.id';
        $st = db()->prepare($sql);
        $st->execute($args);
        $stamp = gmdate('Ymd-His');
        header('Content-Type: text/csv; charset=utf-8');
        header("Content-Disposition: attachment; filename=\"cpi-label-history-$stamp.csv\"");
        $fh = fopen('php://output', 'w');
        fputcsv($fh, ['pid', 'flight', 'mat_file', 'index', 'label', 'made_of', 'flags', 'when_utc', 'username', 'name', 'expertise']);
        foreach ($st as $r) {
            preg_match(PID_RE, $r['pid'], $m);
            fputcsv($fh, [$r['pid'], $m[1] ?? '', $m[2] ?? '', $m[3] ?? '', $r['label'] ?? '(removed)', $r['made_of'], $r['flags'], $r['created_at'],
                          $r['username'], $r['display_name'], $r['expertise']]);
        }
        exit;
    }
    $sql = 'SELECT l.pid, l.flight_id, l.label, l.made_of, l.flags, l.note, l.scheme_version, l.created_at, l.updated_at,
                   u.username, u.display_name, u.expertise, u.affiliation, u.is_external, l.session_id
            FROM labels l JOIN users u ON u.id = l.user_id';
    $args = [];
    if (!$u['is_admin'] || ($_GET['mine'] ?? '') === '1') { $sql .= ' WHERE l.user_id = ?'; $args[] = $u['id']; }
    $sql .= ' ORDER BY l.flight_id, l.pid, u.username';
    $st = db()->prepare($sql);
    $st->execute($args);
    $stamp = gmdate('Ymd-His');
    if ($fmt === 'json') {
        header('Content-Type: application/json; charset=utf-8');
        header("Content-Disposition: attachment; filename=\"cpi-labels-$stamp.json\"");
        echo json_encode($st->fetchAll(), JSON_UNESCAPED_UNICODE | JSON_PRETTY_PRINT);
        exit;
    }
    header('Content-Type: text/csv; charset=utf-8');
    header("Content-Disposition: attachment; filename=\"cpi-labels-$stamp.csv\"");
    $fh = fopen('php://output', 'w');
    fputcsv($fh, ['pid', 'flight', 'mat_file', 'index', 'label', 'made_of', 'flags', 'note', 'scheme_version',
                  'created_utc', 'updated_utc', 'username', 'name', 'expertise', 'affiliation', 'account', 'session_id']);
    foreach ($st as $r) {
        preg_match(PID_RE, $r['pid'], $m);
        fputcsv($fh, [$r['pid'], $r['flight_id'], $m[2] ?? '', $m[3] ?? '', $r['label'], $r['made_of'], $r['flags'], $r['note'],
                      $r['scheme_version'], $r['created_at'], $r['updated_at'], $r['username'], $r['display_name'],
                      $r['expertise'], $r['affiliation'], (int)$r['is_external'] ? 'no login (uploaded)' : 'login', $r['session_id']]);
    }
    exit;

default:
    if (str_starts_with($action, 'admin_')) {
        require __DIR__ . '/admin.php';
        admin_action($action);
    }
    fail(404, 'Unknown action');
}
