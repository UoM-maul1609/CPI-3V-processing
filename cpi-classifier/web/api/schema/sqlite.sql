-- CPI classifier schema for SQLite (testing / small installs).

CREATE TABLE IF NOT EXISTS users (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  username      TEXT NOT NULL UNIQUE,
  display_name  TEXT NOT NULL,
  email         TEXT NULL,
  pass_hash     TEXT NOT NULL,
  created_at    TEXT NOT NULL,
  last_login    TEXT NULL,
  must_change_pw INTEGER NOT NULL DEFAULT 0,
  is_admin      INTEGER NOT NULL DEFAULT 0,
  disabled      INTEGER NOT NULL DEFAULT 0,
  expertise     TEXT NULL,
  affiliation   TEXT NULL,
  is_external   INTEGER NOT NULL DEFAULT 0,
  prefs         TEXT NULL
);

CREATE TABLE IF NOT EXISTS labels (
  id             INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id        INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  pid            TEXT NOT NULL,
  flight_id      TEXT NOT NULL,
  label          TEXT NOT NULL,
  flags          TEXT NOT NULL DEFAULT '',
  note           TEXT NULL,
  scheme_version TEXT NULL,
  created_at     TEXT NOT NULL,
  updated_at     TEXT NOT NULL,
  session_id     INTEGER NULL,
  made_of        TEXT NULL,
  UNIQUE (user_id, pid)
);
CREATE INDEX IF NOT EXISTS ix_pid ON labels(pid);
CREATE INDEX IF NOT EXISTS ix_flight ON labels(flight_id);

CREATE TABLE IF NOT EXISTS label_log (
  id         INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id    INTEGER NOT NULL,
  pid        TEXT NOT NULL,
  label      TEXT NULL,
  flags      TEXT NULL,
  made_of    TEXT NULL,
  created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_log_user ON label_log(user_id);

-- Classifications uploaded from a session file (e.g. made in practice mode).
CREATE TABLE IF NOT EXISTS sessions (
  id             INTEGER PRIMARY KEY AUTOINCREMENT,
  contributor_id INTEGER NOT NULL,
  uploaded_by    INTEGER NOT NULL,
  name           TEXT NULL,
  expertise      TEXT NULL,
  affiliation    TEXT NULL,
  notes          TEXT NULL,
  file_name      TEXT NULL,
  scheme_version TEXT NULL,
  started_utc    TEXT NULL,
  n_labels       INTEGER NOT NULL DEFAULT 0,
  created_at     TEXT NOT NULL
);

-- "Keep me signed in": one row per browser that chose it. Only a hash of the secret is kept.
CREATE TABLE IF NOT EXISTS remember_tokens (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id     INTEGER NOT NULL,
  selector    TEXT NOT NULL UNIQUE,
  token_hash  TEXT NOT NULL,
  created_at  TEXT NOT NULL,
  expires_at  TEXT NOT NULL,
  last_used   TEXT NULL,
  agent       TEXT NULL
);
