<?php
// Project information pages: a description, links and pictures for each project
// (the "campaign" of its flights). Stored as static files next to the flight data,
//   data/projects/<slug>/info.json, data/projects/<slug>/<image files>
//   data/projects/index.json  (a summary of all of them, for the Flights page)
// so the browser reads them directly; PHP only writes them (administrators).

const PROJECT_MAX_IMAGE_BYTES = 25 * 1024 * 1024;
const PROJECT_MAX_IMAGES = 40;
const PROJECT_IMAGE_EDGE = 2000;   // longest side kept, in pixels (when PHP has GD)
const PROJECT_THUMB_EDGE = 560;

function project_slug(string $name): string {
    $s = strtolower(trim(preg_replace('/[^A-Za-z0-9]+/', '-', $name), '-'));
    if ($s === '') $s = 'no-project';
    return substr($s, 0, 60);
}
function projects_dir(): string { return rtrim(data_dir(), '/') . '/projects'; }
function project_dir(string $name, bool $create = false): string {
    $d = projects_dir() . '/' . project_slug($name);
    if ($create && !is_dir($d) && !@mkdir($d, 0775, true)) fail(500, "Can't create $d: check the data folder is writable by the web server");
    return $d;
}
function write_json_atomic(string $path, $data): void {
    $tmp = $path . '.tmp' . getmypid();
    if (file_put_contents($tmp, json_encode($data, JSON_PRETTY_PRINT | JSON_UNESCAPED_SLASHES | JSON_UNESCAPED_UNICODE) . "\n") === false)
        fail(500, "Can't write $path: check the data folder is writable by the web server");
    rename($tmp, $path);
}
function project_info_load(string $name): array {
    $f = project_dir($name) . '/info.json';
    $j = is_file($f) ? json_decode((string)file_get_contents($f), true) : null;
    return is_array($j) ? $j : ['project' => $name, 'title' => '', 'summary' => '', 'body' => '', 'links' => [], 'images' => []];
}
function project_info_store(string $name, array $info, array $u): array {
    $info['project'] = $name;
    $info['updated_utc'] = gmdate('Y-m-d\TH:i:s\Z');
    $info['updated_by'] = $u['display_name'];
    write_json_atomic(project_dir($name, true) . '/info.json', $info);
    project_index_rebuild();
    return $info;
}
function project_index_rebuild(): void {
    $idx = [];
    foreach (glob(projects_dir() . '/*/info.json') ?: [] as $f) {
        $j = json_decode((string)file_get_contents($f), true);
        if (!is_array($j) || !isset($j['project'])) continue;
        $img = $j['images'][0] ?? null;
        $idx[$j['project']] = [
            'slug' => basename(dirname($f)),
            'title' => $j['title'] ?? '',
            'summary' => $j['summary'] ?? '',
            'cover' => $img ? ($img['thumb'] ?? $img['file']) : null,
            'n_images' => count($j['images'] ?? []),
            'n_links' => count($j['links'] ?? []),
            'updated_utc' => $j['updated_utc'] ?? null,
            'flight_notes' => (object)($j['flight_notes'] ?? []),
        ];
    }
    if (!is_dir(projects_dir())) @mkdir(projects_dir(), 0775, true);
    write_json_atomic(projects_dir() . '/index.json', ['version' => 1, 'projects' => $idx]);
}
function clean_text($v, int $max): string {
    $s = str_replace("\r\n", "\n", (string)$v);
    $s = preg_replace('/[^\P{C}\n\t]/u', '', $s) ?? '';
    return mb_substr(trim($s), 0, $max);
}
function clean_url(string $url): string {
    $url = trim($url);
    if ($url === '') return '';
    if (!preg_match('~^[a-z][a-z0-9+.-]*:~i', $url) && preg_match('~^[\w-]+(\.[\w-]+)+(/|$)~', $url)) $url = 'https://' . $url;   // "example.org/x"
    $scheme = strtolower((string)parse_url($url, PHP_URL_SCHEME));
    if (!in_array($scheme, ['http', 'https', 'mailto'], true) || mb_strlen($url) > 1000) fail(400, "Links must start with http://, https:// or mailto: ($url)");
    return $url;
}

/** Save an uploaded picture: rotated upright, shrunk to PROJECT_IMAGE_EDGE, plus a thumbnail. */
function project_store_image(array $file, string $dir): array {
    if (($file['error'] ?? UPLOAD_ERR_NO_FILE) !== UPLOAD_ERR_OK) {
        $why = [UPLOAD_ERR_INI_SIZE => 'it is bigger than PHP allows (upload_max_filesize in php.ini)', UPLOAD_ERR_FORM_SIZE => 'it is too big',
                UPLOAD_ERR_PARTIAL => 'the upload was interrupted', UPLOAD_ERR_NO_FILE => 'no file arrived'][$file['error'] ?? 4] ?? 'upload error ' . ($file['error'] ?? '?');
        fail(400, "The picture couldn’t be uploaded: $why");
    }
    if ($file['size'] > PROJECT_MAX_IMAGE_BYTES) fail(400, 'Pictures can be up to 25 MB');
    $info = @getimagesize($file['tmp_name']);
    $types = [IMAGETYPE_JPEG => 'jpg', IMAGETYPE_PNG => 'png', IMAGETYPE_GIF => 'gif', IMAGETYPE_WEBP => 'webp'];
    if (!$info || !isset($types[$info[2]])) fail(400, 'Use a JPEG, PNG, GIF or WebP picture');
    $ext = $types[$info[2]];
    $base = bin2hex(random_bytes(6));
    $main = "$base.$ext";
    $thumb = "$base.thumb.$ext";
    [$w, $h] = $info;
    $gd = function_exists('imagecreatetruecolor') && $ext !== 'gif';
    if ($gd) {
        $load = ['jpg' => 'imagecreatefromjpeg', 'png' => 'imagecreatefrompng', 'webp' => 'imagecreatefromwebp'][$ext];
        $im = function_exists($load) ? @$load($file['tmp_name']) : false;
        if (!$im) $gd = false;
    }
    if ($gd) {
        // phone photos: turn upright using the EXIF orientation
        if ($ext === 'jpg' && function_exists('exif_read_data')) {
            $o = (int)(@exif_read_data($file['tmp_name'])['Orientation'] ?? 1);
            $rot = [3 => 180, 6 => -90, 8 => 90][$o] ?? 0;
            if ($rot) { $r = imagerotate($im, $rot, 0); if ($r) { imagedestroy($im); $im = $r; } }
        }
        $w = imagesx($im); $h = imagesy($im);
        $save = function ($src, int $edge, string $path) use ($ext) {
            $sw = imagesx($src); $sh = imagesy($src);
            $k = min(1, $edge / max($sw, $sh));
            $dw = max(1, (int)round($sw * $k)); $dh = max(1, (int)round($sh * $k));
            $dst = $src;
            if ($k < 1) {
                $dst = imagecreatetruecolor($dw, $dh);
                if ($ext !== 'jpg') { imagealphablending($dst, false); imagesavealpha($dst, true); }
                imagecopyresampled($dst, $src, 0, 0, 0, 0, $dw, $dh, $sw, $sh);
            }
            if ($ext !== 'jpg') imagesavealpha($dst, true);
            $ok = $ext === 'jpg' ? imagejpeg($dst, $path, 86) : ($ext === 'png' ? imagepng($dst, $path, 6) : imagewebp($dst, $path, 84));
            if ($dst !== $src) imagedestroy($dst);
            if (!$ok) fail(500, "Can't write $path");
            return [$dw, $dh];
        };
        [$w, $h] = $save($im, PROJECT_IMAGE_EDGE, "$dir/$main");
        $save($im, PROJECT_THUMB_EDGE, "$dir/$thumb");
        imagedestroy($im);
    } else {
        // no GD (or a GIF, which may be animated): keep it as it is
        if (!move_uploaded_file($file['tmp_name'], "$dir/$main") && !copy($file['tmp_name'], "$dir/$main")) fail(500, "Can't write to $dir");
        $thumb = $main;
    }
    @chmod("$dir/$main", 0664);
    @chmod("$dir/$thumb", 0664);
    return ['file' => $main, 'thumb' => $thumb, 'w' => $w, 'h' => $h, 'caption' => '', 'credit' => ''];
}

function project_action(string $action, array $u): void {
    switch ($action) {
    case 'admin_project_save':
        // {project, title, summary, body, links: [{title, url, note}], images: [{file, caption, credit}] (new order),
        //  flight_notes: {flight id: text}}
        $b = require_post();
        $name = clean_text($b['project'] ?? '', 120);
        if ($name === '') fail(400, 'Which project?');
        $lock = fopen(project_dir($name, true) . '/.lock', 'c');
        flock($lock, LOCK_EX);
        $info = project_info_load($name);
        $info['title'] = clean_text($b['title'] ?? '', 160);
        $info['summary'] = clean_text($b['summary'] ?? '', 400);
        $info['body'] = clean_text($b['body'] ?? '', 20000);
        $links = [];
        foreach (array_slice((array)($b['links'] ?? []), 0, 60) as $l) {
            $url = clean_url((string)($l['url'] ?? ''));
            if ($url === '') continue;
            $links[] = ['title' => clean_text($l['title'] ?? '', 160) ?: preg_replace('~^(https?://|mailto:)(www\.)?~', '', $url),
                        'url' => $url, 'note' => clean_text($l['note'] ?? '', 300)];
        }
        $info['links'] = $links;
        if (isset($b['flight_notes']) && is_array($b['flight_notes'])) {
            $notes = [];
            foreach ($b['flight_notes'] as $fid => $txt) {
                $fid = (string)$fid;
                $txt = clean_text($txt, 4000);
                if ($txt !== '' && preg_match('/^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$/', $fid)) $notes[$fid] = $txt;
            }
            $info['flight_notes'] = (object)$notes;
        }
        if (isset($b['images']) && is_array($b['images'])) {
            $have = [];
            foreach ($info['images'] ?? [] as $im) $have[$im['file']] = $im;
            $order = [];
            foreach ($b['images'] as $im) {
                $f = (string)($im['file'] ?? '');
                if (!isset($have[$f])) continue;
                $order[] = array_merge($have[$f], ['caption' => clean_text($im['caption'] ?? '', 500), 'credit' => clean_text($im['credit'] ?? '', 200)]);
                unset($have[$f]);
            }
            $info['images'] = array_merge($order, array_values($have));   // anything not mentioned stays, at the end
        }
        $info = project_info_store($name, $info, $u);
        flock($lock, LOCK_UN);
        out(['info' => $info]);

    case 'admin_project_image':
        // multipart upload: project, file
        if ($_SERVER['REQUEST_METHOD'] !== 'POST') fail(405, 'POST required');
        if (($_SERVER['HTTP_X_REQUESTED_WITH'] ?? '') !== 'cpi-classifier') fail(403, 'Bad request origin');
        $name = clean_text($_POST['project'] ?? '', 120);
        if ($name === '') fail(400, 'Which project?');
        if (empty($_FILES['file'])) fail(400, 'No picture arrived. If it is large, check upload_max_filesize and post_max_size in php.ini.');
        $dir = project_dir($name, true);
        $lock = fopen("$dir/.lock", 'c');
        flock($lock, LOCK_EX);
        $info = project_info_load($name);
        if (count($info['images'] ?? []) >= PROJECT_MAX_IMAGES) fail(400, 'That’s the most pictures a project can have (' . PROJECT_MAX_IMAGES . ')');
        $img = project_store_image($_FILES['file'], $dir);
        $img['caption'] = clean_text($_POST['caption'] ?? '', 500);
        $info['images'][] = $img;
        $info = project_info_store($name, $info, $u);
        flock($lock, LOCK_UN);
        out(['image' => $img, 'info' => $info]);

    case 'admin_project_image_delete':
        $b = require_post();
        $name = clean_text($b['project'] ?? '', 120);
        $dir = project_dir($name);
        $lock = fopen(project_dir($name, true) . '/.lock', 'c');
        flock($lock, LOCK_EX);
        $info = project_info_load($name);
        $keep = [];
        foreach ($info['images'] ?? [] as $im) {
            if ($im['file'] === ($b['file'] ?? null)) {
                foreach (array_unique([$im['file'], $im['thumb'] ?? $im['file']]) as $f)
                    if (preg_match('/^[a-f0-9]{12}(\.thumb)?\.(jpg|png|gif|webp)$/', $f) && is_file("$dir/$f")) unlink("$dir/$f");
            } else $keep[] = $im;
        }
        $info['images'] = $keep;
        $info = project_info_store($name, $info, $u);
        flock($lock, LOCK_UN);
        out(['info' => $info]);
    }
    fail(400, 'Unknown action');
}
