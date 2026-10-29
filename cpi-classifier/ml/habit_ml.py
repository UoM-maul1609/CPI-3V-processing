#!/usr/bin/env python3
"""
habit_ml.py - train and run an automatic ice-crystal habit classifier (Keras 3)
on the particles and classifications of the CPI classifier web site.

The web site's Admin page (Machine learning) runs this as a background job; it
can also be run by hand:

    python3 habit_ml.py check                    versions and GPUs, as JSON
    python3 habit_ml.py train   JOB.json         train a model (see train_job below)
    python3 habit_ml.py predict JOB.json         classify particles with a model

The model sees each particle's image (padded to a square, so its shape is kept;
optionally turned so its long axis lies left-right, and optionally kept at the
instrument's pixel size rather than scaled to fill the square: see prepare_image)
together with its measured size and shape (maximum length, width, equivalent
diameter, aspect ratio, area ratio, all in µm where it applies), so it knows how big
the particle is even though every image is resized to the same square. It starts
from an image network trained on everyday photos (EfficientNetV2), and is
fine-tuned on your classifications.

Alongside the class it learns the degree of riming (none / lightly / moderately)
and the other flags people give (hollow, chain, hit by large drops, breaking up, cut
off), as extra outputs of the same network.

Faint, washed-out images (stray light in the camera, seen with some 3V-CPI and CPI
1.5 data) are handled in two ways: each image is stretched so its own background
is black and its darkest part white before the network sees it, and training also
shows it images with their contrast lowered and noise added. The report gives the
accuracy for low-, medium- and high-contrast particles, so you can see whether it
works.

Testing: whole flights are held back (not random particles, which would look
better than they are, as particles from one flight look alike). The report has
accuracy overall, per class, per instrument, a confusion matrix, and the most
confident mistakes.

Lines starting "@progress {json}" are read by the web page for its progress bar.
"""
import csv
import json
import math
import os
import random
import sys
import time
import warnings

warnings.filterwarnings("ignore", message=".*shuffle=True.*")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

FEATURES = ["len", "wid", "deq", "aspect", "area_ratio"]      # logs of these, standardised
LOG_FEATURES = {"len", "wid", "deq", "aspect"}


def progress(**kw):
    print("@progress " + json.dumps(kw), flush=True)


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------- the site's data
def load_flight(data_dir, fid, _cache={}):
    """A flight's particles: stems, file index, ROI number and measurements, as numpy arrays."""
    import numpy as np
    if fid in _cache:
        return _cache[fid]
    fdir = os.path.join(data_dir, "flights", fid)
    with open(os.path.join(fdir, "catalog.json"), encoding="utf-8") as fh:
        c = json.load(fh)
    n = int(c["n"])
    cols = {}
    if "bin" in c:
        with open(os.path.join(fdir, c["bin"]["file"]), "rb") as fh:
            buf = fh.read()
        T = {"u32": "<u4", "f64": "<f8", "f32": "<f4"}
        for k, (t, off) in c["bin"]["cols"].items():
            cols[k] = np.frombuffer(buf, dtype=T[t], count=n, offset=off)
    else:
        for k, v in c["cols"].items():
            cols[k] = np.array([np.nan if x is None else x for x in v], dtype=float)
    meta = {}
    try:
        with open(os.path.join(fdir, "meta.json"), encoding="utf-8") as fh:
            meta = json.load(fh)
    except OSError:
        pass
    stems = c["files"]
    pid_at = {f"{fid}:{stems[int(f)]}:{int(i)}": k for k, (f, i) in enumerate(zip(cols["f"], cols["i"]))}
    out = {"fid": fid, "dir": fdir, "n": n, "stems": stems, "cols": cols, "meta": meta, "pid_at": pid_at}
    _cache[fid] = out
    return out


def instrument_of(meta):
    ins = meta.get("instrument") or {}
    name = f"{ins.get('name') or ''} {ins.get('version') or ''}".strip()
    if "3v" in name.lower().replace("-", "").replace(" ", ""):
        return "3V-CPI"
    v = str(ins.get("version") or "")
    try:
        x = float("".join(ch for ch in v if ch.isdigit() or ch == ".") or "nan")
    except ValueError:
        x = float("nan")
    if x != x:
        return "CPI (version not given)"
    return "CPI v2.0" if x >= 2 else "CPI v1.5" if x >= 1.5 else "CPI v1.0"


def particle_rows(data_dir, pids):
    """For each pid: (image path, raw feature vector, flight id, instrument, contrast); None if not found."""
    import numpy as np
    out = []
    for pid in pids:
        fid = pid.split(":", 1)[0]
        try:
            F = load_flight(data_dir, fid)
        except (OSError, ValueError, KeyError):
            out.append(None); continue
        k = F["pid_at"].get(pid)
        if k is None:
            out.append(None); continue
        stem, idx = F["stems"][int(F["cols"]["f"][k])], int(F["cols"]["i"][k])
        path = os.path.join(F["dir"], "img", stem, f"{idx}.png")
        feats = [float(F["cols"][f][k]) if f in F["cols"] else float("nan") for f in FEATURES]
        con = float(F["cols"]["contrast"][k]) if "contrast" in F["cols"] else float("nan")
        out.append((path, feats, fid, instrument_of(F["meta"]), con))
    return out


def feature_matrix(raw, stats=None):
    """log where it applies, then standardise (missing values become 0 = the average)."""
    import numpy as np
    X = np.array(raw, dtype="float64")
    for j, f in enumerate(FEATURES):
        if f in LOG_FEATURES:
            with np.errstate(divide="ignore", invalid="ignore"):
                X[:, j] = np.where(X[:, j] > 0, np.log(X[:, j]), np.nan)
    if stats is None:
        mean = np.nanmean(X, axis=0); std = np.nanstd(X, axis=0)
        mean = np.where(np.isfinite(mean), mean, 0.0); std = np.where(np.isfinite(std) & (std > 0), std, 1.0)
        stats = {"mean": mean.tolist(), "std": std.tolist()}
    X = (X - np.array(stats["mean"])) / np.array(stats["std"])
    X[~np.isfinite(X)] = 0.0
    return X.astype("float32"), stats


# ---------------------------------------------------------------- images
PREP_DEFAULT = {"orient": False, "scale": "fit", "antialias": False}   # how models before these options saw images


def prep_options(o):
    """The image preparation chosen for a training run (see prepare_image)."""
    return {"orient": bool(o.get("orient", False)),
            "scale": "native" if o.get("scale") == "native" else "fit",
            "antialias": True}


def prepare_image(img, size, prep):
    """img: (h, w, 1) float32, particle bright (up to 255) on a background of 0.
    prep["orient"]: turn the particle so its long axis (from its image moments) lies
      left-right, then crop to it. Near-round particles (minor/major axis > 0.8), and ones that
      fill their picture to its edges, aren't turned.
    prep["scale"]: "fit" scales every particle so its longer side fills the square (small ones
      are blown up); "native" keeps the instrument's pixels (same um per pixel for every particle)
      and only shrinks particles bigger than the square.
    Either way the shape is kept and the rest of the square is background."""
    import tensorflow as tf
    if prep.get("orient"):
        h, w = tf.shape(img)[0], tf.shape(img)[1]
        # a particle filling its picture to the edges (cut off, or a big faint smear) isn't turned:
        # turning it would put the picture's straight edges at a slant, which looks like a shape
        edge = tf.concat([img[0, :, 0], img[-1, :, 0], img[:, 0, 0], img[:, -1, 0]], axis=0)
        at_edge = tf.reduce_mean(tf.cast(edge > 40.0, tf.float32)) > 0.15
        d = tf.cast(tf.math.ceil(tf.sqrt(tf.cast(h * h + w * w, tf.float32))), tf.int32) + 2
        img = tf.image.resize_with_crop_or_pad(img, d, d)             # room to turn without losing corners
        wt = tf.maximum(img[..., 0] - 40.0, 0.0)                      # ignore faint background
        m = tf.reduce_sum(wt) + 1e-6
        ys, xs = tf.meshgrid(tf.range(d, dtype=tf.float32), tf.range(d, dtype=tf.float32), indexing="ij")
        cx, cy = tf.reduce_sum(wt * xs) / m, tf.reduce_sum(wt * ys) / m
        mu20 = tf.reduce_sum(wt * (xs - cx) ** 2) / m
        mu02 = tf.reduce_sum(wt * (ys - cy) ** 2) / m
        mu11 = tf.reduce_sum(wt * (xs - cx) * (ys - cy)) / m
        half, root = (mu20 + mu02) / 2, tf.sqrt(((mu20 - mu02) / 2) ** 2 + mu11 ** 2)
        round_ = (half - root) / tf.maximum(half + root, 1e-6) > 0.64  # axis ratio > 0.8
        theta = tf.where(round_ | at_edge, 0.0, 0.5 * tf.atan2(2 * mu11, mu20 - mu02))   # long axis angle, y down
        c, s_ = tf.cos(theta), tf.sin(theta)
        ctr = (tf.cast(d, tf.float32) - 1) / 2
        # output pixel (x, y) takes the input at the particle's centroid + R(theta) (x - ctr, y - ctr)
        t = tf.stack([c, -s_, cx - c * ctr + s_ * ctr, s_, c, cy - s_ * ctr - c * ctr, 0.0, 0.0])
        img = tf.raw_ops.ImageProjectiveTransformV3(
            images=img[None], transforms=t[None], output_shape=tf.stack([d, d]), fill_value=0.0,
            interpolation="BILINEAR", fill_mode="CONSTANT")[0]
        on = tf.reduce_max(img[..., 0], axis=1) > 40.0, tf.reduce_max(img[..., 0], axis=0) > 40.0
        rows, cols = tf.where(on[0])[:, 0], tf.where(on[1])[:, 0]

        def crop():
            y0 = tf.cast(tf.maximum(tf.reduce_min(rows) - 2, 0), tf.int32)
            x0 = tf.cast(tf.maximum(tf.reduce_min(cols) - 2, 0), tf.int32)
            y1 = tf.cast(tf.minimum(tf.reduce_max(rows) + 3, tf.cast(d, tf.int64)), tf.int32)
            x1 = tf.cast(tf.minimum(tf.reduce_max(cols) + 3, tf.cast(d, tf.int64)), tf.int32)
            return img[y0:y1, x0:x1]
        img = tf.cond(tf.size(rows) > 0, crop, lambda: img)
    if not prep.get("antialias"):                                     # as models before the options
        return tf.image.resize_with_pad(img, size, size)
    h, w = tf.cast(tf.shape(img)[0], tf.float32), tf.cast(tf.shape(img)[1], tf.float32)
    k = size / tf.maximum(h, w)
    if prep.get("scale") == "native":
        k = tf.minimum(k, 1.0)
    nh = tf.clip_by_value(tf.cast(tf.round(h * k), tf.int32), 1, size)
    nw = tf.clip_by_value(tf.cast(tf.round(w * k), tf.int32), 1, size)
    img = tf.image.resize(img, tf.stack([nh, nw]), antialias=True)
    return tf.image.resize_with_crop_or_pad(img, size, size)


def load_image(path, size, prep):
    """Read a particle PNG and prepare it as the network sees it: (size, size, 3), 0-255."""
    import tensorflow as tf
    raw = tf.io.read_file(path)
    img = tf.io.decode_png(raw, channels=3)                     # grey, palette or colour PNGs alike
    img = tf.cast(tf.image.rgb_to_grayscale(img), tf.float32)
    # stretch each image by its own levels: background (95th percentile) -> 0, darkest part -> 255,
    # so a washed-out image (stray light) looks like a crisp one. The particle ends up bright on a
    # dark background, so padding and rotation fill with "background".
    v = tf.sort(tf.reshape(img, [-1]))
    n = tf.shape(v)[0]
    bg = v[tf.cast(tf.cast(n - 1, tf.float32) * 0.95, tf.int32)]
    dk = v[tf.cast(tf.cast(n - 1, tf.float32) * 0.005, tf.int32)]
    span = bg - dk
    img = tf.where(span > 8.0, tf.clip_by_value((bg - img) / tf.maximum(span, 1.0), 0.0, 1.0) * 255.0, 255.0 - img)
    img = prepare_image(img, size, prep)
    img = tf.ensure_shape(img, [size, size, 1])
    return tf.image.grayscale_to_rgb(img)


def make_dataset(paths, feats, labels, size, batch, shuffle=False, seed=0, weights=None, prep=None):
    """labels: an array, or a dict of arrays (one per output). weights: per-particle weights (training).
    prep: image preparation (prep_options); None = as models made before the options."""
    import tensorflow as tf
    prep = prep or PREP_DEFAULT

    def load(path, f, y, *w):
        x = {"image": load_image(path, size, prep), "features": f}
        return (x, y, w[0]) if w else (x, y)

    cols = (list(paths), feats, labels) if weights is None else (list(paths), feats, labels, weights)
    ds = tf.data.Dataset.from_tensor_slices(cols)
    if shuffle:
        ds = ds.shuffle(min(len(paths), 20000), seed=seed, reshuffle_each_iteration=True)
    return ds.map(load, num_parallel_calls=tf.data.AUTOTUNE).batch(batch).prefetch(tf.data.AUTOTUNE)


def save_seen(paths, size, prep, out_png, n=32):
    """A sheet of particles as the network sees them (before the random training changes)."""
    import tensorflow as tf
    imgs = [load_image(p, size, prep)[..., :1] for p in paths[:n]]
    if not imgs:
        return
    cols = 8
    while len(imgs) % cols:
        imgs.append(tf.zeros_like(imgs[0]))
    rows = [tf.concat(imgs[r:r + cols], axis=1) for r in range(0, len(imgs), cols)]
    sheet = 255.0 - tf.concat(rows, axis=0)                      # shown dark on light, like the site
    tf.io.write_file(out_png, tf.io.encode_png(tf.cast(tf.clip_by_value(sheet, 0, 255), tf.uint8)))


def build_model(n_classes, n_feats, size, base_name, weights, n_riming=0, n_flags=0, orient=False):
    import keras
    from keras import layers
    img_in = keras.Input((size, size, 3), name="image")
    f_in = keras.Input((n_feats,), name="features")
    x = layers.RandomFlip("horizontal_and_vertical")(img_in)
    # any angle; or, when particles are already turned to their long axis, only a little either way
    x = layers.RandomRotation(0.03 if orient else 0.5, fill_mode="constant", fill_value=0.0)(x)
    x = layers.RandomContrast(0.35)(x)                        # images with less contrast than usual
    x = layers.Rescaling(1 / 255.0)(x)
    x = layers.GaussianNoise(0.025)(x)                        # and more noise, as with stray light
    x = layers.Rescaling(255.0)(x)
    bases = {"EfficientNetV2B0": keras.applications.EfficientNetV2B0, "EfficientNetV2B1": keras.applications.EfficientNetV2B1,
             "EfficientNetV2S": keras.applications.EfficientNetV2S}
    make = bases.get(base_name, keras.applications.EfficientNetV2B0)
    used = weights
    try:
        base = make(include_top=False, weights=weights, input_shape=(size, size, 3), pooling="avg")
    except Exception as e:                                    # noqa: BLE001  (no internet for the weights)
        log(f"WARNING: couldn't load pretrained {base_name} weights ({e}); training from scratch, which needs many more labels")
        base = make(include_top=False, weights=None, input_shape=(size, size, 3), pooling="avg")
        used = None
    base._name = "backbone"
    # a pretrained network always runs its BatchNormalization layers in inference mode, even while
    # it is being fine-tuned: otherwise the small batches of particle images overwrite the statistics
    # it learnt, and the test accuracy collapses as soon as fine-tuning starts
    x = base(x, training=False) if used else base(x)
    g = layers.Dense(32, activation="relu")(f_in)
    x = layers.Concatenate()([x, g])
    x = layers.Dropout(0.3)(x)
    outs = {"habit": layers.Dense(n_classes, activation="softmax", dtype="float32", name="habit")(x)}
    if n_riming:
        outs["riming"] = layers.Dense(n_riming, activation="softmax", dtype="float32", name="riming")(x)
    if n_flags:
        outs["flags"] = layers.Dense(n_flags, activation="sigmoid", dtype="float32", name="flags")(x)
    return keras.Model({"image": img_in, "features": f_in}, outs), base, used


class Progress:
    """Keras callback: one @progress line per epoch for the web page."""
    def __init__(self, total, offset=0, stage="training"):
        import keras
        self.cb = keras.callbacks.LambdaCallback(on_epoch_begin=self.begin, on_epoch_end=self.end, on_train_batch_end=self.batch)
        self.total, self.offset, self.stage, self.t0, self.last, self.epoch = total, offset, stage, time.time(), 0, offset

    def begin(self, epoch, logs):
        self.epoch = self.offset + epoch

    def batch(self, b, logs):
        if time.time() - self.last > 5:
            self.last = time.time()
            progress(stage=self.stage, epoch=self.epoch + 1, epochs=self.total, batch=b + 1)

    def end(self, epoch, logs):
        logs = logs or {}
        e = self.offset + epoch + 1
        msg = {k: round(float(v), 4) for k, v in logs.items() if isinstance(v, (int, float))}
        for a, b in (("accuracy", "habit_accuracy"), ("val_accuracy", "val_habit_accuracy")):
            if b in msg and a not in msg:
                msg[a] = msg[b]
        log(f"epoch {e}/{self.total}: " + ", ".join(f"{k} {v}" for k, v in msg.items()))
        progress(stage=self.stage, epoch=e, epochs=self.total, **msg, seconds=round(time.time() - self.t0))


# ---------------------------------------------------------------- train
def train_job(job):
    """job = {data_dir, labels_csv, out_dir, classes: [ids in order], flag_list: [ids], riming_levels: [names],
    options: {img_size, epochs, base, batch, min_examples, val_fraction, seed, weights}}.
    labels_csv columns: pid,label[,riming (0 = none, 1, 2 ...)][,flags (ids joined by ;)]"""
    import numpy as np
    import keras
    import tensorflow as tf

    o = job.get("options", {})
    size, epochs, batch = int(o.get("img_size", 160)), int(o.get("epochs", 15)), int(o.get("batch", 64))
    seed = int(o.get("seed", 1))
    random.seed(seed); np.random.seed(seed); keras.utils.set_random_seed(seed)
    out = job["out_dir"]
    os.makedirs(out, exist_ok=True)
    gpus = tf.config.list_physical_devices("GPU")
    log(f"TensorFlow {tf.__version__}, Keras {keras.__version__}; GPU: {', '.join(g.name for g in gpus) or 'none (CPU only: slow)'}")
    if gpus:
        keras.mixed_precision.set_global_policy("mixed_float16")

    progress(stage="reading", done=0, total=0)
    with open(job["labels_csv"], newline="", encoding="utf-8") as fh:
        lab = [(r["pid"], r["label"], int(r.get("riming") or 0), [f for f in (r.get("flags") or "").split(";") if f]) for r in csv.DictReader(fh)]
    rows = particle_rows(job["data_dir"], [x[0] for x in lab])
    items = [(x, r) for x, r in zip(lab, rows) if r is not None and os.path.exists(r[0])]
    log(f"{len(lab):,} classified particles; {len(items):,} with images on the site")

    # classes with enough examples (others are left out and reported)
    counts = {}
    for x, _ in items:
        counts[x[1]] = counts.get(x[1], 0) + 1
    order = [c for c in job.get("classes", []) if c in counts] + sorted(c for c in counts if c not in job.get("classes", []))
    min_n = int(o.get("min_examples", 20))
    classes = [c for c in order if counts[c] >= min_n]
    dropped = {c: counts[c] for c in order if counts[c] < min_n}
    if dropped:
        log("Left out (fewer than %d examples): %s" % (min_n, ", ".join(f"{c} {n}" for c, n in dropped.items())))
    if len(classes) < 2:
        raise SystemExit(f"Need at least two classes with {min_n} or more examples; have: " + ", ".join(f"{c} {n}" for c, n in counts.items()))
    cix = {c: i for i, c in enumerate(classes)}
    items = [it for it in items if it[0][1] in cix]

    # riming degree and flags: learnt only when people have used them enough to learn from
    levels = job.get("riming_levels") or []
    rim_n = np.bincount([min(x[2], len(levels) - 1) if levels else 0 for x, _ in items], minlength=max(1, len(levels)))
    use_riming = len(levels) > 1 and (rim_n[1:] >= 10).any()
    flag_list = [f for f in (job.get("flag_list") or []) if sum(f in x[3] for x, _ in items) >= 10]
    skipped_flags = [f for f in (job.get("flag_list") or []) if f not in flag_list]
    log(f"Also learning: " + ", ".join((["riming degree"] if use_riming else []) + flag_list) if (use_riming or flag_list) else "Not enough riming or flags given to learn them yet")

    # hold back whole flights for testing (about val_fraction of the particles)
    by_flight = {}
    for k, it in enumerate(items):
        by_flight.setdefault(it[1][2], []).append(k)
    flights = sorted(by_flight)
    rnd = random.Random(seed)
    frac = float(o.get("val_fraction", 0.2))
    val_idx, val_flights = [], []
    if len(flights) >= 3:
        rnd.shuffle(flights)
        target = frac * len(items)
        for f in flights:
            if len(val_idx) >= target or len(val_flights) >= len(flights) - 1:
                break
            val_flights.append(f); val_idx += by_flight[f]
        split = "whole flights held back"
    else:
        idx = list(range(len(items))); rnd.shuffle(idx)
        val_idx = idx[: max(1, int(frac * len(items)))]
        split = "random particles held back (fewer than 3 flights: the test will look better than it really is)"
    vset = set(val_idx)
    tr = [it for k, it in enumerate(items) if k not in vset]
    va = [it for k, it in enumerate(items) if k in vset]
    missing = [c for c in classes if not any(it[0][1] == c for it in tr)]
    if missing:
        log("WARNING: no training examples left for " + ", ".join(missing) + " after holding flights back")
    log(f"Training on {len(tr):,} particles, testing on {len(va):,} ({split}); {len(classes)} classes")

    def targets(its):
        y = {"habit": np.array([cix[x[1]] for x, _ in its], dtype="int32")}
        if use_riming:
            y["riming"] = np.array([min(x[2], len(levels) - 1) for x, _ in its], dtype="int32")
        if flag_list:
            y["flags"] = np.array([[1.0 if f in x[3] else 0.0 for f in flag_list] for x, _ in its], dtype="float32")
        return y

    Xtr, stats = feature_matrix([r[1] for _, r in tr])
    Xva, _ = feature_matrix([r[1] for _, r in va], stats)
    ytr, yva = targets(tr), targets(va)

    # rare classes count for more (inverse square root of their frequency), as a weight per particle
    freq = np.bincount(ytr["habit"], minlength=len(classes)).astype(float)
    cw = np.where(freq > 0, 1.0 / np.sqrt(np.maximum(freq, 1)), 0.0)
    cw = cw / (cw[freq > 0].mean() if (freq > 0).any() else 1)
    sw = cw[ytr["habit"]].astype("float32")
    prep = prep_options(o)
    log(f"Images: {'turned to their long axis, ' if prep['orient'] else ''}"
        + ("instrument pixels kept (only big particles shrunk)" if prep["scale"] == "native" else "each scaled to fill the square"))
    ds_tr = make_dataset([r[0] for _, r in tr], Xtr, ytr, size, batch, shuffle=True, seed=seed, weights=sw, prep=prep)
    ds_va = make_dataset([r[0] for _, r in va], Xva, yva, size, batch, prep=prep)
    try:
        save_seen([r[0] for _, r in random.Random(seed).sample(tr, min(32, len(tr)))], size, prep, os.path.join(out, "seen.png"))
    except Exception as e:                                    # noqa: BLE001
        log(f"(couldn't save the sample sheet: {e})")

    weights = o.get("weights", "imagenet") or None           # downloaded once into $KERAS_HOME/models
    model, base, used = build_model(len(classes), len(FEATURES), size, o.get("base", "EfficientNetV2B0"), weights,
                                    len(levels) if use_riming else 0, len(flag_list), prep["orient"])
    head_epochs = min(5, max(2, epochs // 4)) if used else 0
    total = head_epochs + epochs
    hist = {}
    losses = {"habit": "sparse_categorical_crossentropy"}
    lw = {"habit": 1.0}
    mets = {"habit": ["accuracy", keras.metrics.SparseTopKCategoricalAccuracy(2, name="top2")]}
    if use_riming:
        losses["riming"] = "sparse_categorical_crossentropy"; lw["riming"] = 0.5; mets["riming"] = ["accuracy"]
    if flag_list:
        losses["flags"] = "binary_crossentropy"; lw["flags"] = 0.5; mets["flags"] = [keras.metrics.BinaryAccuracy(name="accuracy")]

    # keep the epoch with the best test accuracy for the class (what the report gives; the loss can
    # keep falling while the accuracy falls). With only the class to learn, Keras names it plain "accuracy"
    watch = "val_habit_accuracy" if (use_riming or flag_list) else "val_accuracy"

    def fit(n, lr, offset, trainable):
        base.trainable = trainable
        if trainable and used:                                # keep BatchNormalization frozen (see build_model)
            for layer in base.layers:
                if isinstance(layer, keras.layers.BatchNormalization):
                    layer.trainable = False
        model.compile(optimizer=keras.optimizers.Adam(lr), loss=losses, loss_weights=lw, metrics=mets)
        cbs = [Progress(total, offset).cb]
        if trainable:
            cbs.append(keras.callbacks.EarlyStopping(monitor=watch, mode="max", patience=4, restore_best_weights=True))
            cbs.append(keras.callbacks.ReduceLROnPlateau(monitor=watch, mode="max", factor=0.3, patience=2))
        h = model.fit(ds_tr, validation_data=ds_va, epochs=n, callbacks=cbs, verbose=0)
        for k, v in h.history.items():
            hist.setdefault(k, []).extend(float(x) for x in v)

    t0 = time.time()
    if head_epochs:
        log(f"First {head_epochs} epochs: only the new layers learn (the pretrained network stays as it is)")
        fit(head_epochs, 1e-3, 0, False)
        head_best, head_w = max(hist.get(watch, [-1.0])), model.get_weights()
        log("Then the whole network is fine-tuned, gently")
    fit(epochs, 1e-5 if used else 1e-3, head_epochs, True)
    if head_epochs:
        tuned_best = max(hist.get(watch, [-1.0])[head_epochs:] or [-1.0])
        if tuned_best < head_best:                            # fine-tuning made it worse on the test flights
            model.set_weights(head_w)
            log("Fine-tuning didn't improve on the test flights, so the version from the first epochs is kept")
    train_seconds = round(time.time() - t0)
    if "val_habit_accuracy" in hist and "val_accuracy" not in hist:
        hist["val_accuracy"] = hist["val_habit_accuracy"]

    # ------------------------------------------------ test
    progress(stage="testing", done=0, total=len(va))
    nC = len(classes)
    if len(va):
        out_va = model.predict(ds_va, verbose=0)
        P = out_va["habit"] if isinstance(out_va, dict) else out_va
    else:
        out_va, P = {}, np.zeros((0, nC))
    yh = yva["habit"]
    pred = P.argmax(1) if len(va) else np.zeros(0, int)
    cm = np.zeros((nC, nC), int)
    for t, p in zip(yh, pred):
        cm[t, p] += 1
    per_class = []
    for i, c in enumerate(classes):
        tp = int(cm[i, i]); sup = int(cm[i].sum()); pp = int(cm[:, i].sum())
        prec = tp / pp if pp else None; rec = tp / sup if sup else None
        f1 = 2 * prec * rec / (prec + rec) if prec and rec else (0.0 if sup else None)
        per_class.append({"class": c, "train": int(freq[i]), "test": sup, "precision": prec, "recall": rec, "f1": f1})
    acc = float((pred == yh).mean()) if len(va) else None
    top2 = float(np.mean([yh[k] in np.argsort(P[k])[-2:] for k in range(len(va))])) if len(va) else None
    by_inst = {}
    for k, (_, r) in enumerate(va):
        b = by_inst.setdefault(r[3], [0, 0]); b[0] += int(pred[k] == yh[k]); b[1] += 1
    # accuracy for low-, medium- and high-contrast particles (thirds of the test set)
    by_contrast = []
    con = np.array([r[4] for _, r in va], dtype=float)
    ok = np.isfinite(con)
    if ok.sum() >= 30:
        q = np.quantile(con[ok], [1 / 3, 2 / 3])
        for name, lo, hi in (("low contrast", -np.inf, q[0]), ("medium contrast", q[0], q[1]), ("high contrast", q[1], np.inf)):
            m = ok & (con > lo) & (con <= hi) if np.isfinite(lo) else ok & (con <= hi)
            if m.sum():
                by_contrast.append({"band": name, "from": None if not np.isfinite(lo) else float(lo), "to": None if not np.isfinite(hi) else float(hi),
                                    "n": int(m.sum()), "accuracy": float((pred[m] == yh[m]).mean())})
    riming_report = flags_report = None
    if use_riming and len(va):
        R = out_va["riming"]; rp = R.argmax(1); rt = yva["riming"]
        rcm = np.zeros((len(levels), len(levels)), int)
        for t, p in zip(rt, rp):
            rcm[t, p] += 1
        riming_report = {"levels": levels, "accuracy": float((rp == rt).mean()), "confusion": rcm.tolist()}
    if flag_list and len(va):
        Fp = out_va["flags"] >= 0.5; Ft = yva["flags"] >= 0.5
        flags_report = []
        for j, f in enumerate(flag_list):
            tp = int((Fp[:, j] & Ft[:, j]).sum()); pp = int(Fp[:, j].sum()); sup = int(Ft[:, j].sum())
            flags_report.append({"flag": f, "test": sup, "precision": tp / pp if pp else None, "recall": tp / sup if sup else None})
    mistakes = sorted(((float(P[k, pred[k]]), x[0], classes[yh[k]], classes[pred[k]]) for k, (x, _) in enumerate(va) if pred[k] != yh[k]), reverse=True)[:60]
    f1s = [x["f1"] for x in per_class if x["f1"] is not None]

    model.save(os.path.join(out, "model.keras"))
    info = {
        "format": 2, "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "classes": classes, "dropped_classes": dropped, "img_size": size, "features": FEATURES, "feature_stats": stats,
        "riming_levels": levels if use_riming else [], "flag_list": flag_list, "flags_not_learnt": skipped_flags,
        "preprocess": "per-image stretch", "prep": prep, "base": o.get("base", "EfficientNetV2B0"), "pretrained": used or "none", "options": o,
        "train_flights": sorted(set(flights) - set(val_flights)) if val_flights else flights, "test_flights": val_flights, "split": split,
        "n_train": len(tr), "n_test": len(va), "train_seconds": train_seconds, "epochs_run": len(hist.get("loss", [])),
        "metrics": {"accuracy": acc, "top2": top2, "macro_f1": (sum(f1s) / len(f1s)) if f1s else None,
                    "per_class": per_class, "confusion": cm.tolist(),
                    "by_instrument": {k: {"accuracy": v[0] / v[1], "n": v[1]} for k, v in by_inst.items()},
                    "by_contrast": by_contrast, "riming": riming_report, "flags": flags_report},
        "history": hist,
        "mistakes": [{"pid": p, "true": t, "pred": q, "prob": round(pr, 3)} for pr, p, t, q in mistakes],
        "gpu": [g.name for g in gpus], "tensorflow": tf.__version__, "keras": keras.__version__,
    }
    with open(os.path.join(out, "model.json"), "w") as fh:
        json.dump(info, fh, indent=1)
    log(f"Test accuracy {acc:.1%} (top 2: {top2:.1%}), mean F1 {info['metrics']['macro_f1']:.2f}" if acc is not None else "No test particles")
    progress(stage="done", accuracy=acc)


# ---------------------------------------------------------------- predict
def predict_job(job):
    """job = {data_dir, model_dir, out_csv, flights: [ids], pids: [ids] (optional), batch}.
    Writes pid,flight_id,label,prob,label2,prob2,riming,riming_prob,flags for every particle with an image."""
    import numpy as np
    import keras
    import tensorflow as tf
    with open(os.path.join(job["model_dir"], "model.json")) as fh:
        info = json.load(fh)
    model = keras.models.load_model(os.path.join(job["model_dir"], "model.keras"), compile=False)
    classes, size, batch = info["classes"], int(info["img_size"]), int(job.get("batch", 256))
    levels, flag_list = info.get("riming_levels") or [], info.get("flag_list") or []
    prep = info.get("prep") or PREP_DEFAULT                   # exactly as the model was trained
    gpus = tf.config.list_physical_devices("GPU")
    log(f"Model with {len(classes)} classes; GPU: {', '.join(g.name for g in gpus) or 'none (CPU only)'}")

    work = []                                                 # (flight id, [pids])
    if job.get("pids"):
        byf = {}
        for p in job["pids"]:
            byf.setdefault(p.split(":", 1)[0], []).append(p)
        work = sorted(byf.items())
    for fid in job.get("flights", []):
        try:
            F = load_flight(job["data_dir"], fid)
        except (OSError, ValueError, KeyError) as e:
            log(f"  {fid}: can't read its particle list in {os.path.join(job['data_dir'], 'flights', fid)} ({e}); skipped")
            continue
        work.append((fid, list(F["pid_at"].keys())))
    total = sum(len(p) for _, p in work)
    log(f"{total:,} particles in {len(work)} flight(s)")
    done = written = 0
    tmp = job["out_csv"] + ".part"
    with open(tmp, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["pid", "flight_id", "label", "prob", "label2", "prob2", "riming", "riming_prob", "flags"])
        for fid, pids in work:
            progress(stage="reading", flight=fid, done=done, total=total)
            rows = particle_rows(job["data_dir"], pids)
            keep = [(p, r) for p, r in zip(pids, rows) if r is not None and os.path.exists(r[0])]
            unknown = sum(r is None for r in rows)
            noimg = len(pids) - len(keep) - unknown
            if unknown or noimg:
                eg = next((r[0] for r in rows if r is not None and not os.path.exists(r[0])), None)
                log(f"  {fid}: {noimg:,} particles without an image" + (f" (e.g. {eg})" if eg else "")
                    + (f", {unknown:,} not in the flight's particle list" if unknown else ""))
            if not keep:
                log(f"  {fid}: nothing to classify"); done += len(pids); continue
            X, _ = feature_matrix([r[1] for _, r in keep], info["feature_stats"])
            for s in range(0, len(keep), 20000):                # in chunks, so millions of particles fit in memory
                chunk = keep[s:s + 20000]
                ds = make_dataset([r[0] for _, r in chunk], X[s:s + 20000], np.zeros(len(chunk), "int32"), size, batch, prep=prep)
                o = model.predict(ds, verbose=0)
                P = o["habit"] if isinstance(o, dict) else o
                top = np.argsort(-P, axis=1)[:, :2]
                R = o.get("riming") if isinstance(o, dict) else None
                Fl = o.get("flags") if isinstance(o, dict) else None
                for k, ((p, _), t, pr) in enumerate(zip(chunk, top, P)):
                    rim = rimp = ""
                    if R is not None and levels:
                        ri = int(R[k].argmax()); rim = levels[ri] if ri > 0 else ""; rimp = f"{R[k, ri]:.3f}"
                    fl = ";".join(f for j, f in enumerate(flag_list) if Fl is not None and Fl[k, j] >= 0.5)
                    w.writerow([p, fid, classes[t[0]], f"{pr[t[0]]:.4f}", classes[t[1]] if len(classes) > 1 else "",
                                f"{pr[t[1]]:.4f}" if len(classes) > 1 else "", rim, rimp, fl])
                written += len(chunk)
                progress(stage="classifying", flight=fid, done=done + s + len(chunk), total=total)
            done += len(pids)
            log(f"  {fid}: {len(keep):,} particles")
    if not written:
        os.remove(tmp)
        log(f"Classified nothing: none of the {total:,} particles had an image the model could read (see above)."
            " Check the flights' img folders exist and the web server can read them.")
        sys.exit(1)
    os.replace(tmp, job["out_csv"])
    log(f"Classified {written:,} particles")
    progress(stage="done", done=written, total=total)


def check():
    info = {"python": sys.version.split()[0]}
    try:
        import tensorflow as tf
        import keras
        info.update(tensorflow=tf.__version__, keras=keras.__version__,
                    gpus=[g.name for g in tf.config.list_physical_devices("GPU")],
                    cuda=tf.sysconfig.get_build_info().get("cuda_version"))
    except Exception as e:                                    # noqa: BLE001
        info["error"] = f"{type(e).__name__}: {e}"
    print(json.dumps(info))


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("check", "train", "predict"):
        sys.exit(__doc__)
    if sys.argv[1] == "check":
        return check()
    with open(sys.argv[2]) as fh:
        job = json.load(fh)
    (train_job if sys.argv[1] == "train" else predict_job)(job)


if __name__ == "__main__":
    main()
