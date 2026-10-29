-- CPI classifier schema for MySQL / MariaDB (utf8mb4).
-- The API creates these automatically if they are missing; you can also run
-- this file by hand:  mysql -u root -p cpi_classifier < mysql.sql

CREATE TABLE IF NOT EXISTS users (
  id            INT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
  username      VARCHAR(32)  NOT NULL UNIQUE,
  display_name  VARCHAR(80)  NOT NULL,
  email         VARCHAR(190) NULL,
  pass_hash     VARCHAR(255) NOT NULL,
  created_at    DATETIME     NOT NULL,
  last_login    DATETIME     NULL,
  must_change_pw TINYINT     NOT NULL DEFAULT 0,  -- 1 = signed in with a temporary password
  is_admin      TINYINT      NOT NULL DEFAULT 0,
  disabled      TINYINT      NOT NULL DEFAULT 0,
  expertise     VARCHAR(60)  NULL,
  affiliation   VARCHAR(120) NULL,
  is_external   TINYINT      NOT NULL DEFAULT 0,    -- 1 = a contributor without a login (uploaded session)
  prefs         TEXT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- One current classification per person per particle.
CREATE TABLE IF NOT EXISTS labels (
  id             BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
  user_id        INT UNSIGNED NOT NULL,
  pid            VARCHAR(200) NOT NULL,          -- "<flight>:<mat stem>:<index>"
  flight_id      VARCHAR(64)  NOT NULL,
  label          VARCHAR(64)  NOT NULL,
  flags          VARCHAR(255) NOT NULL DEFAULT '', -- comma separated
  note           VARCHAR(500) NULL,
  scheme_version VARCHAR(32)  NULL,
  created_at     DATETIME     NOT NULL,
  updated_at     DATETIME     NOT NULL,
  session_id     INT UNSIGNED NULL,               -- set when the label came from an uploaded session file
  made_of        VARCHAR(32)  NULL,               -- secondary class: what it is made of (e.g. aggregate of plates)
  UNIQUE KEY uq_user_pid (user_id, pid),
  KEY ix_pid (pid),
  KEY ix_flight (flight_id),
  CONSTRAINT fk_labels_user FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Append-only history of every change (label NULL = removed / undone).
CREATE TABLE IF NOT EXISTS label_log (
  id         BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
  user_id    INT UNSIGNED NOT NULL,
  pid        VARCHAR(200) NOT NULL,
  label      VARCHAR(64)  NULL,
  flags      VARCHAR(255) NULL,
  made_of    VARCHAR(32)  NULL,
  created_at DATETIME     NOT NULL,
  KEY ix_log_user (user_id),
  KEY ix_log_pid (pid)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Classifications uploaded from a session file (e.g. made in practice mode).
CREATE TABLE IF NOT EXISTS sessions (
  id             INT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
  contributor_id INT UNSIGNED NOT NULL,
  uploaded_by    INT UNSIGNED NOT NULL,
  name           VARCHAR(80)  NULL,
  expertise      VARCHAR(60)  NULL,
  affiliation    VARCHAR(120) NULL,
  notes          VARCHAR(1000) NULL,
  file_name      VARCHAR(200) NULL,
  scheme_version VARCHAR(32)  NULL,
  started_utc    VARCHAR(32)  NULL,
  n_labels       INT          NOT NULL DEFAULT 0,
  created_at     DATETIME     NOT NULL,
  KEY ix_sess_contrib (contributor_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- "Keep me signed in": one row per browser that chose it. Only a hash of the secret is kept.
CREATE TABLE IF NOT EXISTS remember_tokens (
  id          INT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
  user_id     INT UNSIGNED NOT NULL,
  selector    CHAR(24)     NOT NULL UNIQUE,
  token_hash  CHAR(64)     NOT NULL,
  created_at  DATETIME     NOT NULL,
  expires_at  DATETIME     NOT NULL,
  last_used   DATETIME     NULL,
  agent       VARCHAR(200) NULL,
  KEY ix_rt_user (user_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
