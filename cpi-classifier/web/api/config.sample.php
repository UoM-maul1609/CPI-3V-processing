<?php
// Copy this file to config.php and edit it. config.php is not served as text
// (PHP runs it), but keep it out of version control because it holds passwords.

return [
    // --- Database -------------------------------------------------------
    // MySQL / MariaDB:
    'dsn'      => 'mysql:host=localhost;dbname=cpi_classifier;charset=utf8mb4',
    'db_user'  => 'cpi_web',
    'db_pass'  => 'change-me',
    // SQLite (handy for testing on a laptop; the folder must be writable by
    // the web server and should be OUTSIDE the web root):
    // 'dsn' => 'sqlite:/var/lib/cpi-classifier/labels.sqlite',

    // --- Demonstration site -------------------------------------------------
    // true = no database needed and nothing is saved: everyone can browse and
    // try classifying, and their classifications stay in their own browser.
    // Use it for a public demo copy of the site (it can share the same data folder).
    'demo' => false,

    // --- Accounts -------------------------------------------------------
    // Administrator usernames. The first account created on the site must be one
    // of these (it's offered on the first visit). More administrators can be
    // made from the Admin > People page.
    'admins' => ['pconnolly'],

    // Normally the administrator adds people (Admin > People) and gives them a
    // temporary password. Set true to also let people create their own accounts.
    'allow_registration' => false,

    // If set, people must type this code to register (share it with your group).
    // Leave as '' to let anyone register.
    'registration_code' => '',

    // --- Importing flights from the admin page -----------------------------
    // Folder on this server that holds your MAT files (any layout; one folder
    // per flight is easiest: <project>/<flight>/). The web server only needs to READ it.
    'raw_root' => '/mnt/CPI-raw',
    // Folder the web server can WRITE, outside the web root: saved flight
    // settings and import logs go here.
    'work_dir' => '/srv/cpi/work',
    // Where imported images and stats go (the page reads them from here).
    'data_dir' => __DIR__ . '/../data',
    // Python with numpy + scipy (+ netCDF4 for .nc navigation files).
    'python'   => '/usr/bin/python3',
    // The import script (cpi_export.py) and its colormaps.json.
    'exporter' => __DIR__ . '/../../exporter/cpi_export.py',
    // How many imports may run at once (each loads one MAT file into memory).
    'max_jobs' => 1,

    // --- Machine learning (Admin → Machine learning; see deploy/DEPLOY.md section 15)
    // Python with TensorFlow + Keras 3 (and the GPU) for ml/habit_ml.py.
    // 'ml_python' => '/var/www/cpi/venv-ml/bin/python',
    // PHP on the command line, to load each run's results into the database.
    // 'php_cli'   => '/usr/bin/php',

    // --- Misc -----------------------------------------------------------
    // Choices for 'level of expertise' (people's profiles and uploaded sessions).
    'expertise_levels' => ['New to ice crystal images', 'Some experience', 'Experienced', 'Expert'],

    'session_name' => 'cpiclass',
    // Classification scheme the server validates labels against.
    'scheme_file' => __DIR__ . '/../config/scheme.json',
];
