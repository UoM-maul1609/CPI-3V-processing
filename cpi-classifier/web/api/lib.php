<?php
declare(strict_types=1);

// Runs on PHP 7.4 (Ubuntu 20.04's own package) and PHP 8.x.
if (!function_exists('str_starts_with')) {
    function str_starts_with(string $h, string $n): bool { return $n === '' || strncmp($h, $n, strlen($n)) === 0; }
    function str_ends_with(string $h, string $n): bool { return $n === '' || substr($h, -strlen($n)) === $n; }
    function str_contains(string $h, string $n): bool { return $n === '' || strpos($h, $n) !== false; }
}

function cfg(): array {
    static $c = null;
    if ($c === null) {
        $f = __DIR__ . '/config.php';
        if (!is_file($f)) fail(500, 'Server not configured: copy api/config.sample.php to api/config.php');
        $c = require $f;
    }
    return $c;
}

function fail(int $code, string $msg): void {   // never returns
    http_response_code($code);
    header('Content-Type: application/json; charset=utf-8');
    echo json_encode(['ok' => false, 'error' => $msg]);
    exit;
}

function out($data): void {   // never returns
    header('Content-Type: application/json; charset=utf-8');
    header('Cache-Control: no-store');
    echo json_encode(['ok' => true] + $data, JSON_UNESCAPED_UNICODE);
    exit;
}

function db(): PDO {
    static $pdo = null;
    if ($pdo) return $pdo;
    $c = cfg();
    try {
        $pdo = new PDO($c['dsn'], $c['db_user'] ?? null, $c['db_pass'] ?? null, [
            PDO::ATTR_ERRMODE => PDO::ERRMODE_EXCEPTION,
            PDO::ATTR_DEFAULT_FETCH_MODE => PDO::FETCH_ASSOC,
        ]);
    } catch (PDOException $e) {
        error_log('cpi-classifier db: ' . $e->getMessage());
        fail(500, 'Database connection failed');
    }
    if (is_sqlite()) {
        $pdo->exec('PRAGMA journal_mode=WAL');
        $pdo->exec('PRAGMA foreign_keys=ON');
    }
    return $pdo;
}

function is_sqlite(): bool {
    return str_starts_with(cfg()['dsn'], 'sqlite:');
}

/** Demo mode: the site works without a database and nothing is saved. */
function is_demo(): bool {
    return !empty(cfg()['demo']);
}

/** Create tables if they are missing, and add columns introduced later (idempotent). */
function ensure_schema(): void {
    static $done = false;
    if ($done) return;
    $done = true;
    $file = __DIR__ . '/schema/' . (is_sqlite() ? 'sqlite.sql' : 'mysql.sql');
    if (!is_file($file)) return;
    $sql = preg_replace('/--[^\n]*/', '', file_get_contents($file));   // drop SQL comments
    foreach (array_filter(array_map('trim', explode(';', $sql))) as $stmt) db()->exec($stmt);
    // columns added after the first release
    $cols = [
        ['users', 'must_change_pw', 'INTEGER NOT NULL DEFAULT 0'], ['users', 'is_admin', 'INTEGER NOT NULL DEFAULT 0'],
        ['users', 'disabled', 'INTEGER NOT NULL DEFAULT 0'], ['users', 'expertise', 'VARCHAR(60) NULL'],
        ['users', 'affiliation', 'VARCHAR(120) NULL'], ['users', 'is_external', 'INTEGER NOT NULL DEFAULT 0'],
        ['labels', 'session_id', 'INTEGER NULL'],
        ['users', 'prefs', 'TEXT NULL'],
        ['labels', 'made_of', 'VARCHAR(32) NULL'], ['label_log', 'made_of', 'VARCHAR(32) NULL'],
    ];
    foreach ($cols as [$table, $col, $type]) {
        try { db()->query("SELECT $col FROM $table LIMIT 1"); }
        catch (PDOException $e) { db()->exec("ALTER TABLE $table ADD COLUMN $col $type"); }
    }
}

/** The choices offered for 'level of expertise' (config 'expertise_levels'). */
function expertise_levels(): array {
    return cfg()['expertise_levels'] ?? ['New to ice crystal images', 'Some experience', 'Experienced', 'Expert'];
}

function user_count(): int {
    return (int)db()->query('SELECT COUNT(*) FROM users WHERE is_external = 0')->fetchColumn();
}

/** A temporary password that's easy to read out or type: e.g. kdw7-m3pq-tx9h */
function temp_password(): string {
    $abc = 'abcdefghjkmnpqrstuvwxyz23456789';
    $s = '';
    for ($i = 0; $i < 12; $i++) $s .= $abc[random_int(0, strlen($abc) - 1)];
    return implode('-', str_split($s, 4));
}

/** Check a new password; returns an error message or null. */
function password_problem(string $p, string $username = ''): ?string {
    if (strlen($p) < 10) return 'Use at least 10 characters';
    if ($username !== '' && stripos($p, $username) !== false) return 'Don’t include your username in your password';
    if (preg_match('/^(.)\1+$/', $p)) return 'That password is too easy to guess';
    return null;
}

function start_session(): void {
    if (session_status() === PHP_SESSION_ACTIVE) return;
    session_name(cfg()['session_name'] ?? 'cpiclass');
    session_set_cookie_params([
        'lifetime' => 60 * 60 * 24 * 30,
        'path' => '/',
        'secure' => !empty($_SERVER['HTTPS']) && $_SERVER['HTTPS'] !== 'off',
        'httponly' => true,
        'samesite' => 'Lax',
    ]);
    session_start();
}

// ---------------------------------------------------------- "keep me signed in"
// A long-lived cookie "<selector>:<secret>". The database keeps only a SHA-256 of the
// secret, so a copy of the database can't be used to sign in. Signing out, changing
// your password, or an administrator resetting it or switching the account off ends it.
const REMEMBER_COOKIE = 'cpiremember';
const REMEMBER_DAYS = 365;

function remember_cookie(string $value, int $expires): void {
    setcookie(REMEMBER_COOKIE, $value, [
        'expires' => $expires, 'path' => '/', 'httponly' => true, 'samesite' => 'Lax',
        'secure' => !empty($_SERVER['HTTPS']) && $_SERVER['HTTPS'] !== 'off',
    ]);
}
/** After signing in with "keep me signed in": a new token for this browser. */
function remember_issue(int $uid): void {
    ensure_schema();
    $sel = bin2hex(random_bytes(12));
    $secret = bin2hex(random_bytes(32));
    $exp = time() + REMEMBER_DAYS * 86400;
    db()->prepare('INSERT INTO remember_tokens (user_id, selector, token_hash, created_at, expires_at, agent) VALUES (?,?,?,?,?,?)')
        ->execute([$uid, $sel, hash('sha256', $secret), now_sql(), gmdate('Y-m-d H:i:s', $exp), mb_substr((string)($_SERVER['HTTP_USER_AGENT'] ?? ''), 0, 200)]);
    remember_cookie("$sel:$secret", $exp);
}
/** Forget this browser's token (signing out). */
function remember_forget(): void {
    $c = (string)($_COOKIE[REMEMBER_COOKIE] ?? '');
    if ($c !== '' && strpos($c, ':') !== false) {
        ensure_schema();
        db()->prepare('DELETE FROM remember_tokens WHERE selector = ?')->execute([explode(':', $c, 2)[0]]);
    }
    if ($c !== '') remember_cookie('', time() - 3600);
}
/** Forget every browser for this person (password changed or reset, account switched off). */
function remember_forget_user(int $uid): void {
    ensure_schema();
    db()->prepare('DELETE FROM remember_tokens WHERE user_id = ?')->execute([$uid]);
}
/** No session, but a valid token: sign the person straight back in. */
function remember_resume(): void {
    $c = (string)($_COOKIE[REMEMBER_COOKIE] ?? '');
    if ($c === '' || !preg_match('/^([0-9a-f]{24}):([0-9a-f]{64})$/', $c, $m)) return;
    ensure_schema();
    $st = db()->prepare('SELECT r.id, r.user_id, r.token_hash, r.expires_at, r.last_used, u.disabled FROM remember_tokens r JOIN users u ON u.id = r.user_id WHERE r.selector = ?');
    $st->execute([$m[1]]);
    $r = $st->fetch();
    if (!$r || !hash_equals((string)$r['token_hash'], hash('sha256', $m[2])) || strcmp((string)$r['expires_at'], now_sql()) < 0 || (int)$r['disabled']) {
        if ($r && strcmp((string)$r['expires_at'], now_sql()) < 0) db()->prepare('DELETE FROM remember_tokens WHERE id = ?')->execute([$r['id']]);
        remember_cookie('', time() - 3600);
        return;
    }
    session_regenerate_id(true);
    $_SESSION['uid'] = (int)$r['user_id'];
    // keep it going while it's used: push the expiry on, at most once a day
    if (!$r['last_used'] || strcmp((string)$r['last_used'], gmdate('Y-m-d H:i:s', time() - 86400)) < 0) {
        $exp = time() + REMEMBER_DAYS * 86400;
        db()->prepare('UPDATE remember_tokens SET last_used = ?, expires_at = ? WHERE id = ?')->execute([now_sql(), gmdate('Y-m-d H:i:s', $exp), $r['id']]);
        db()->prepare('UPDATE users SET last_login = ? WHERE id = ?')->execute([now_sql(), $r['user_id']]);
        remember_cookie($c, $exp);
    }
}

function current_user(bool $fresh = false): ?array {
    start_session();
    if (empty($_SESSION['uid'])) remember_resume();
    if (empty($_SESSION['uid'])) return null;
    static $u = false;
    if ($fresh) $u = false;
    if ($u === false) {
        ensure_schema();
        $st = db()->prepare('SELECT id, username, display_name, email, must_change_pw, is_admin, disabled, expertise, affiliation, prefs FROM users WHERE id = ?');
        $st->execute([$_SESSION['uid']]);
        $u = $st->fetch() ?: null;
        if ($u && (int)$u['disabled']) $u = null;
        if ($u) {
            $u['id'] = (int)$u['id'];
            $u['is_admin'] = (int)$u['is_admin'] === 1 || in_array($u['username'], cfg()['admins'] ?? [], true);
            $u['must_change_pw'] = (int)$u['must_change_pw'] === 1;
            $u['prefs'] = $u['prefs'] ? (json_decode($u['prefs'], true) ?: (object)[]) : (object)[];   // display settings, so they follow you between computers
            unset($u['disabled']);
        }
    }
    return $u;
}

function require_user(bool $allowTemp = false): array {
    if (is_demo()) fail(403, 'This is a demonstration site: nothing is saved');
    $u = current_user();
    if (!$u) fail(401, 'Please sign in');
    if ($u['must_change_pw'] && !$allowTemp) fail(403, 'Please choose your own password first');
    return $u;
}

function require_post(): array {
    if ($_SERVER['REQUEST_METHOD'] !== 'POST') fail(405, 'POST required');
    // Same-origin check: the app always sends this header. A cross-site form
    // cannot set it, and a cross-site fetch with it needs CORS, which we don't allow.
    if (($_SERVER['HTTP_X_REQUESTED_WITH'] ?? '') !== 'cpi-classifier') fail(403, 'Bad request origin');
    $body = json_decode(file_get_contents('php://input') ?: '{}', true);
    if (!is_array($body)) fail(400, 'Invalid JSON');
    return $body;
}

/** What a class can be "made of" (secondary classification): class id => [allowed values]. */
function made_of_map(array $sch): array {
    $m = [];
    foreach ($sch['classes'] ?? [] as $c) if (!empty($c['of']) && is_array($c['of'])) $m[$c['id']] = array_map('strval', $c['of']);
    return $m;
}
/** A valid made_of value for this label, or null. */
function clean_made_of(array $map, string $label, $v): ?string {
    if ($v === null || $v === '') return null;
    $v = (string)$v;
    return isset($map[$label]) && in_array($v, $map[$label], true) ? $v : null;
}

/** flag id -> the classes it applies to (flags with no list apply to every class). */
function flag_for_map(array $sch): array {
    $m = [];
    foreach ($sch['flags'] ?? [] as $f) if (!empty($f['for']) && is_array($f['for'])) $m[$f['id']] = array_map('strval', $f['for']);
    return $m;
}
/** Drop flags that don't apply to this label (e.g. Chain on a plate). */
function flag_applies(array $forMap, string $flag, string $label): bool {
    return !isset($forMap[$flag]) || in_array($label, $forMap[$flag], true);
}

/** The classes and flags: the admin's edited copy if there is one, else config/scheme.json. */
function scheme_override_file(): ?string {
    $w = cfg()['work_dir'] ?? '';
    return $w ? rtrim($w, '/') . '/scheme.json' : null;
}
function scheme(): array {
    static $s = null;
    if ($s === null) {
        $o = scheme_override_file();
        if ($o && is_file($o)) $s = json_decode((string)file_get_contents($o), true);
        if (!$s) {
            $f = cfg()['scheme_file'] ?? (__DIR__ . '/../config/scheme.json');
            $s = json_decode((string)@file_get_contents($f), true) ?: ['classes' => [], 'flags' => []];
        }
    }
    return $s;
}

const PID_RE = '/^([A-Za-z0-9][A-Za-z0-9_.-]{0,63}):([A-Za-z0-9_.-]{1,128}):(\d{1,9})$/';

function now_sql(): string { return gmdate('Y-m-d H:i:s'); }
