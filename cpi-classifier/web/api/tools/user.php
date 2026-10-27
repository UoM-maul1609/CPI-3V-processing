<?php
// Command-line account tool, for the server's terminal only.
//
//   php web/api/tools/user.php add pconnolly "Paul Connolly" --admin
//   php web/api/tools/user.php reset pconnolly
//   php web/api/tools/user.php list
//
// 'add' and 'reset' print a temporary password; the person must choose their
// own the first time they sign in.
declare(strict_types=1);
if (PHP_SAPI !== 'cli') { http_response_code(404); exit; }
require __DIR__ . '/../lib.php';

function say(string $s): void { fwrite(STDOUT, $s . "\n"); }
$args = array_slice($argv, 1);
$cmd = $args[0] ?? '';
ensure_schema();
switch ($cmd) {
case 'add':
    [$username, $name] = [strtolower($args[1] ?? ''), $args[2] ?? ''];
    if (!preg_match('/^[a-z0-9][a-z0-9_.-]{2,31}$/', $username) || $name === '') { say('usage: user.php add USERNAME "Full Name" [--admin]'); exit(1); }
    $st = db()->prepare('SELECT 1 FROM users WHERE username = ?');
    $st->execute([$username]);
    if ($st->fetch()) { say("$username already exists (use reset)"); exit(1); }
    $temp = temp_password();
    db()->prepare('INSERT INTO users (username, display_name, pass_hash, created_at, is_admin, must_change_pw, disabled) VALUES (?,?,?,?,?,1,0)')
        ->execute([$username, $name, password_hash($temp, PASSWORD_DEFAULT), now_sql(), in_array('--admin', $args, true) ? 1 : 0]);
    say("Added $username ($name)" . (in_array('--admin', $args, true) ? ' as an administrator' : '') . ".");
    say("Temporary password: $temp   (they'll be asked to choose their own when they sign in)");
    break;
case 'reset':
    $username = strtolower($args[1] ?? '');
    $temp = temp_password();
    $st = db()->prepare('UPDATE users SET pass_hash = ?, must_change_pw = 1, disabled = 0, is_external = 0 WHERE username = ?');
    $st->execute([password_hash($temp, PASSWORD_DEFAULT), $username]);
    if (!$st->rowCount()) { say("No user $username"); exit(1); }
    say("New temporary password for $username: $temp");
    break;
case 'list':
    foreach (db()->query('SELECT username, display_name, is_admin, disabled, must_change_pw, is_external, last_login FROM users ORDER BY username') as $r)
        say(sprintf('%-20s %-28s %s%s%s%s', $r['username'], $r['display_name'], $r['is_admin'] ? '[admin] ' : '',
            $r['disabled'] ? '[off] ' : '', $r['must_change_pw'] ? '[temporary password] ' : '', $r['is_external'] ? '(uploaded sessions only, no login)' : 'last sign-in ' . ($r['last_login'] ?: 'never')));
    break;
default:
    say("usage:\n  php user.php add USERNAME \"Full Name\" [--admin]\n  php user.php reset USERNAME\n  php user.php list");
    exit(1);
}
