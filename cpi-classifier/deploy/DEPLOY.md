# Deploying the classifier on mccikpc2-PowerEdge-R7525 (Ubuntu 20.04)

The plan: Apache serves the page at `http://<server>/cpi/`, PHP stores accounts
and classifications in MariaDB (or MySQL), and the importer runs in its own
Python environment. While ports 80/443 are closed to the outside, you and
colleagues reach it through an SSH tunnel (step 9).

| What | Where |
|---|---|
| Code (your cpi3v repo; the classifier in its `cpi-classifier/` folder) | `/var/www/cpi/cpi3v` |
| Images, catalogues, project pages (written by imports and the Admin page) | `/var/www/cpi/data` |
| Import logs and the edited class list | `/var/www/cpi/work` |
| Python for the importer | `/var/www/cpi/venv` |
| Your MAT files / pipeline exports (read only) | `/mnt/CPI-data` |
| Settings and database password | `/var/www/cpi/cpi3v/cpi-classifier/web/api/config.php` (not in git) |

Commands are run as you, using `sudo` where shown.

---

## 0. PHP on Ubuntu 20.04, and its security

Ubuntu 20.04 ships **PHP 7.4**. PHP's own developers stopped fixing 7.4 in
November 2022, and Ubuntu's free security updates for 20.04 ended in May 2025.
The usual source of PHP 8 for Ubuntu (Ondřej Surý's PPA) doesn't build for
20.04 any more either. So PHP 7.4 on this machine is only safe if:

* **The machine is attached to Ubuntu Pro (ESM).** Canonical still patches
  php7.4 for 20.04 there, until 2030. The PHP security notices of 2025 and
  October 2026 all have "fix available with Ubuntu Pro" for 20.04. Check with:
  ```bash
  pro status                     # "esm-infra  yes  enabled" is what you want
  apt policy php7.4-common       # an installed version ending in +esmN = patched
  ```
  If it isn't attached, ask IT (the University may have a licence), or use a
  free personal token (up to 5 machines): `sudo pro attach <token>`, then
  `sudo apt update && sudo apt upgrade`.
* **Until then, nobody but you can reach it.** Step 9 makes Apache listen on
  the server itself only; you get in through an SSH tunnel. Nothing on the
  network can talk to PHP, whatever its version.

**For the long term, and before IT opens 80/443,** get onto a supported
release. Either:

* **Upgrade Ubuntu** to 22.04 (PHP 8.1) and then 24.04 (PHP 8.3) with
  `do-release-upgrade`, or ask IT to. 20.04 itself is out of standard support,
  so IT will probably want this anyway. Afterwards, install the `php8.x-*`
  versions of the packages in step 2. Nothing in the classifier changes.
* **Or run PHP in a container** (official `php:8.3-apache` image, updated by
  pulling a new image) while the host stays on 20.04. Ask, and I'll add a
  ready-made Dockerfile.

The classifier runs on PHP 7.4 and 8.x, so you can start now and move later
without changing it.

## 1. See what's already there

```bash
lsb_release -d; php -v; apache2 -v; mysql --version
systemctl is-active apache2 nginx mysql mariadb
sudo ss -tlnp | grep -E ':(80|443|3306) '
pro status 2>/dev/null | head -5
```

If **nginx** is already using port 80, tell me and I'll give you the nginx +
php-fpm version instead. If **MySQL** is already installed and used by
something else, keep it and skip the MariaDB install in step 2.

## 2. Install the packages

```bash
sudo apt update
sudo apt install apache2 libapache2-mod-php php-mysql php-mbstring php-gd \
                 mariadb-server python3-venv git
sudo a2enmod headers expires deflate
```

* **`php-*`:** these install the 7.4 versions on 20.04. `php-gd` resizes
  project photos; `php-mbstring` is required.
* **`mariadb-server`:** skip it if MySQL is already installed.

Check PHP has what it needs:

```bash
php -m | grep -E 'pdo_mysql|mbstring|gd|json'      # all four should be listed
```

## 3. The database

```bash
sudo mysql
```
```sql
CREATE DATABASE cpi_classifier CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
CREATE USER 'cpi_web'@'localhost' IDENTIFIED BY 'choose-a-long-password';
GRANT ALL PRIVILEGES ON cpi_classifier.* TO 'cpi_web'@'localhost';
FLUSH PRIVILEGES;
EXIT;
```

The tables are created automatically on first use, and new columns are added
automatically when you update the code.

## 4. Folders and permissions

Everything lives in `/var/www/cpi`, beside the usual web root. Only the page
itself appears in `/var/www/html` (as `/var/www/html/cpi`, a link made in
step 5), so the repository's `.git`, Python code and `config.php` are never
inside the web root.

Apache runs as `www-data`. It must be able to write the data and work folders
(imports from the Admin page, project pictures). You must be able to as well,
for imports from the command line. A shared group does both:

```bash
sudo groupadd cpi
sudo usermod -aG cpi $USER
sudo usermod -aG cpi www-data
sudo mkdir -p /var/www/cpi/data /var/www/cpi/work
sudo chown -R $USER:cpi /var/www/cpi
sudo chmod 2775 /var/www/cpi /var/www/cpi/data /var/www/cpi/work     # 2 = new files keep the cpi group
```

Make files that Apache creates group-writable too, so imports from the page
and from your own command line can update each other's files:

```bash
echo 'umask 002' | sudo tee -a /etc/apache2/envvars
```

Log out and back in, so your new group takes effect (`id` should list `cpi`).

The MAT files only need to be readable by Apache:

```bash
sudo chgrp -R cpi /mnt/CPI-data && sudo chmod -R g+rX /mnt/CPI-data
```

If `/mnt/CPI-data` is a mount, add `gid=cpi` (CIFS) or set the group on the
NFS side instead.

## 5. The code

Clone your repository into `/var/www/cpi/cpi3v`:

```bash
git clone git@github.com:<you>/cpi3v.git /var/www/cpi/cpi3v
#  or, without SSH keys on the server:  git clone https://github.com/<you>/cpi3v.git /var/www/cpi/cpi3v
#  (for a private repo, GitHub asks for a personal access token instead of a password)
ls /var/www/cpi/cpi3v/cpi-classifier/web/index.html            # check it's there
```

Put the page in the web root as a link:

```bash
sudo ln -s /var/www/cpi/cpi3v/cpi-classifier/web /var/www/html/cpi
```

Don't clone the whole repository into `/var/www/html`: everything in it,
including `.git` (your full history) and the Python code, would be downloadable.

No repo yet? Unzip `cpi-classifier.zip` and move its contents to
`/var/www/cpi/cpi3v/cpi-classifier`. The layout is the same.

## 6. Python for the importer

The Admin page runs the importer as `www-data`. It needs a Python that
`www-data` can use. An environment in your home folder or in conda is often not
readable by `www-data`, so make one in `/var/www/cpi`:

```bash
python3 -m venv /var/www/cpi/venv
/var/www/cpi/venv/bin/pip install --upgrade pip
/var/www/cpi/venv/bin/pip install numpy scipy        # + netCDF4 if you have netCDF navigation files
/var/www/cpi/venv/bin/python /var/www/cpi/cpi3v/cpi-classifier/exporter/cpi_export.py --help | head -3
```

On 20.04 this is Python 3.8. pip picks the newest numpy/scipy that support it,
and the importer works with those.

## 7. Settings: config.php

```bash
cd /var/www/cpi/cpi3v/cpi-classifier/web/api
cp config.sample.php config.php
nano config.php
```

Set these:

```php
'dsn'      => 'mysql:host=localhost;dbname=cpi_classifier;charset=utf8mb4',
'db_user'  => 'cpi_web',
'db_pass'  => 'choose-a-long-password',      // from step 3
'admins'   => ['pconnolly'],                  // the username(s) you'll sign in with as administrator
'raw_root' => '/mnt/CPI-data',
'work_dir' => '/var/www/cpi/work',
'data_dir' => '/var/www/cpi/data',
'python'   => '/var/www/cpi/venv/bin/python',
'exporter' => '/var/www/cpi/cpi3v/cpi-classifier/exporter/cpi_export.py',
```

Then make it readable by Apache, but not by other users of the machine:

```bash
sudo chown $USER:www-data config.php && chmod 640 config.php
```

## 8. Apache

```bash
sudo cp /var/www/cpi/cpi3v/cpi-classifier/deploy/apache-cpi-classifier.conf /etc/apache2/conf-available/cpi-classifier.conf
sudo a2enconf cpi-classifier
sudo apache2ctl configtest          # "Syntax OK"
sudo systemctl reload apache2
curl -s 'http://localhost/cpi/api/?a=me'      # {"ok":true,"user":null,...}
```

What that file does:

* **The page:** `/var/www/html/cpi` is served at `/cpi/` and `/var/www/cpi/data` at
  `/cpi/data/`, so the data stays outside the code folder. Other paths? Change
  the two `Define` lines at the top of the file.
* **Uploads can't run:** nothing in the data folder is ever run as code.
* **Private files stay private:** the schema files and command-line tools can't
  be fetched over the web.
* **Picture size:** project pictures can be up to 25 MB.

If `curl` gives an error, `sudo tail /var/log/apache2/error.log` says why.
Usually it's `config.php` (database password) or folder permissions.

## 9. Using it while 80/443 are closed

First, make Apache listen on the server itself only, so no other machine can
reach it (see step 0). In `/etc/apache2/ports.conf`, change `Listen 80` to
`Listen 127.0.0.1:80`. If there is a `Listen 443` inside `<IfModule ssl_module>`,
comment it out. Then:

```bash
sudo systemctl restart apache2
sudo ss -tlnp | grep apache        # should show 127.0.0.1:80 only
```

If other people already use web pages on this server, skip this. Instead, put
`Require local` in place of `Require all granted` in both `<Directory>` blocks of
`cpi-classifier.conf`. Then only `/cpi/` is restricted to the server itself.

Then, from your own computer, open an SSH tunnel to the server and browse through it:

```bash
ssh -N -L 8080:localhost:80 mccikpc2@<server address>
```

Then open **http://localhost:8080/cpi/** in your browser. Leave the ssh
command running; Ctrl-C closes the tunnel. Colleagues need an account on the
server to do the same. If IT will open a port to the campus network only (say
8080), that avoids the tunnel. Ask them, and I'll add a `Listen 8080` virtual
host.

The local firewall, if it's on (`sudo ufw status`), doesn't affect the tunnel.

## 9b. Letting people on campus in (no tunnel)

Once PHP is patched (Ubuntu Pro, step 0), you can let people on the campus
network use it directly, while everyone else stays shut out:

1. **Listen on the network again.** In `/etc/apache2/ports.conf`, put back
   `Listen 80`.
2. **Campus only.** In `/etc/apache2/conf-available/cpi-classifier.conf`,
   replace both `Require all granted` lines with:
   ```
   Require ip 130.88.0.0/16 10.0.0.0/8 127.0.0.1 ::1
   ```
   The default Ubuntu page at `/` stays visible, which is harmless.
3. **Restart and check:**
   ```bash
   sudo apache2ctl configtest && sudo systemctl restart apache2
   sudo ss -tlnp | grep apache2         # *:80
   ```
4. **Test from another campus computer:** open `http://<server name or IP>/cpi/`.
   - **Nothing at all (times out):** the network blocks port 80 even inside
     campus. Ask IT to allow it internally, or use the tunnel meanwhile.
   - **"Forbidden":** that computer's address isn't in the list. Find it with
     `sudo tail -5 /var/log/apache2/error.log` (look for `client denied ... <address>`),
     then add its range to the `Require ip` lines.
5. **Passwords:** without HTTPS they cross the network unencrypted. That's
   acceptable on campus for a short while, but ask IT for a certificate (step 13)
   before relying on it.

## 10. First run

1. **Administrator account:** open the page, click *Set up the administrator
   account*, and choose a password. The username must be one of `admins` in
   config.php.
2. **Admin → Flights and imports:** the top line should say *Server ready to
   import (Python 3.8…)*. If it doesn't, it says what's missing.
3. **Import:** your folders under `/mnt/CPI-data` are listed there. Import one.
4. **People:** add people on **Admin → People**. Each gets a temporary password.
5. **Project pages:** open a project from **Flights** and use *Edit page* to
   add its description, links, pictures and flight notes.

## 11. Updating

```bash
cd /var/www/cpi/cpi3v && git pull
```

That's all. The page checks for new code on each load (a normal refresh is
enough), and the database gains any new columns by itself. Your `config.php`,
data and work folders are untouched by git.

## 12. Back-ups

Imported images can always be made again from the MAT files. The things you
can't recreate are the classifications, the project pages and the edited class
list:

```bash
sudo mkdir -p /var/www/cpi/backup && sudo chown $USER:cpi /var/www/cpi/backup
crontab -e
```
```
# every night at 02:17: database + project pages + class list, kept for 30 days
17 2 * * * mysqldump --single-transaction -u cpi_web -p'choose-a-long-password' cpi_classifier | gzip > /var/www/cpi/backup/db-$(date +\%F).sql.gz && tar czf /var/www/cpi/backup/pages-$(date +\%F).tgz -C /var/www/cpi data/projects work/scheme.json 2>/dev/null; find /var/www/cpi/backup -mtime +30 -delete
```

The command above puts the password on the command line. To avoid that, put it
in `~/.my.cnf` instead (`[mysqldump]` section, `chmod 600`) and drop `-p…`.

## 13. When IT opens 80 and 443

* **Supported PHP first:** be on a supported PHP before this (step 0), and put
  `Listen 80` back.

* **Name and certificate:** ask IT for a DNS name (e.g.
  `cpi.<dept>.manchester.ac.uk`) and either a certificate from them or
  permission to use Let's Encrypt. For Let's Encrypt:
  `sudo apt install certbot python3-certbot-apache && sudo certbot --apache -d <name>`.
  Port 80 has to be reachable from the internet for this.
* **Secure sign-in cookies:** once the page is served over HTTPS, the sign-in
  cookie is marked secure automatically.
* **Campus only, at first:** in `cpi-classifier.conf`, replace
  `Require all granted` with
  `Require ip 130.88.0.0/16 10.0.0.0/8 127.0.0.1 ::1`.
* **Demonstration copy:** for a public, read-only copy, use a copy of `web/`
  whose `config.php` has `'demo' => true`. Give it two more aliases:
  `/cpi-demo/data` → `/var/www/cpi/data` (so it shares the data) and
  `/cpi-demo` → that copy.

## 14. After upgrading Ubuntu (20.04 → 22.04 → 24.04)

Do the upgrade one release at a time (`sudo do-release-upgrade`), with a backup
from step 12 taken first. Then, for the classifier:

```bash
# PHP: swap Apache's PHP module for the new version (8.1 on 22.04, 8.3 on 24.04)
sudo apt install libapache2-mod-php php-mysql php-mbstring php-gd
sudo a2dismod php7.4; sudo a2enmod php8.3          # use the version you now have: ls /etc/apache2/mods-available/php*
sudo systemctl restart apache2

# Python: the venv points at the old Python (3.8), so make it again
rm -rf /var/www/cpi/venv
python3 -m venv /var/www/cpi/venv && /var/www/cpi/venv/bin/pip install numpy scipy

# check
curl -s 'http://localhost/cpi/api/?a=me'
```

Then **Admin → Flights and imports** should say *Server ready to import*.
MariaDB upgrades itself during the release upgrade, and the classifications,
data and config.php are untouched. Also check `/etc/apache2/envvars` still ends
with `umask 002`: the upgrade may offer to replace that file.

## 15. Machine learning (GPU)

Admin → **Machine learning** trains habit classifiers on the GPU and runs them over
flights. It runs `ml/habit_ml.py` (Keras 3 on TensorFlow) as the web server's user,
in its own Python environment, separate from the importer's.

**1. Python 3.11 for the web server.** TensorFlow needs Python 3.9 or newer; Ubuntu
20.04's own is 3.8. The deadsnakes packages give a 3.11 every user can run:
```bash
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt install python3.11 python3.11-venv
```

**2. TensorFlow with GPU support**, in `/var/www/cpi/venv-ml`:
```bash
sudo python3.11 -m venv /var/www/cpi/venv-ml
sudo /var/www/cpi/venv-ml/bin/pip install --upgrade pip
sudo /var/www/cpi/venv-ml/bin/pip install "tensorflow[and-cuda]>=2.16" "keras>=3.0" numpy
```
The `[and-cuda]` extra brings the CUDA libraries; the NVIDIA driver must be 525 or newer
(`nvidia-smi` shows it).

**3. Check, as the web server:**
```bash
ls -l /dev/nvidia*          # usually crw-rw-rw-: every user can use the GPU
sudo -u www-data /var/www/cpi/venv-ml/bin/python /var/www/cpi/cpi3v/cpi-classifier/ml/habit_ml.py check
```
It prints the versions and `"gpus": ["/physical_device:GPU:0"]`. An empty list means
TensorFlow can't see the GPU: the lines above it say why.

**4. config.php:**
```php
'ml_python' => '/var/www/cpi/venv-ml/bin/python',
'php_cli'   => '/usr/bin/php',          // loads each run's results into the database
```

**5. Pretrained weights.** Training starts from a network trained on everyday photos,
downloaded once (about 25 MB) into `work_dir/ml/keras`. If the server can't reach
storage.googleapis.com as www-data, fetch them once yourself:
```bash
sudo -u www-data env KERAS_HOME=/var/www/cpi/work/ml/keras /var/www/cpi/venv-ml/bin/python -c \
  "import keras; keras.applications.EfficientNetV2B0(include_top=False, weights='imagenet')"
```
(use your `work_dir`). Without them it trains from scratch, which needs many more
classified particles.

Then press **Check the GPU** on Admin → Machine learning.
