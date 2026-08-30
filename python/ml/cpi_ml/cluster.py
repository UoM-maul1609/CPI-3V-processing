"""Run Deep Embedded Clustering from a trained CPI autoencoder."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
from sklearn.cluster import KMeans

from .clustering import clustering_layer_class, target_distribution
from .data import load_batch
from .models import encoder_from_autoencoder


def _predict_all(model, dataset, n, kind, batch_size):
    output = None
    for low in range(0, n, batch_size):
        high = min(low + batch_size, n)
        x = load_batch(dataset, np.arange(low, high), kind)
        values = np.asarray(model.predict(x, verbose=0))
        if output is None:
            output = np.empty((n, values.shape[1]), dtype=np.float32)
        output[low:high] = values
    return output


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("autoencoder")
    parser.add_argument("output_model")
    parser.add_argument("output_encoding")
    parser.add_argument("--kind", choices=["image", "properties"], default="image")
    parser.add_argument("--clusters", type=int, default=9)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--max-iterations", type=int, default=8000)
    parser.add_argument("--update-interval", type=int, default=60)
    parser.add_argument("--tolerance", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--freeze-encoder", action="store_true")
    args = parser.parse_args(argv)

    import keras

    keras.utils.set_random_seed(args.seed)
    autoencoder = keras.saving.load_model(args.autoencoder, compile=False)
    encoder = encoder_from_autoencoder(autoencoder)
    if args.freeze_encoder:
        encoder.trainable = False

    with h5py.File(args.dataset, "r") as handle:
        n = len(handle["lens"])

    embedding = _predict_all(encoder, args.dataset, n, args.kind, args.batch_size)
    kmeans = KMeans(n_clusters=args.clusters, n_init=20, random_state=args.seed)
    labels = kmeans.fit_predict(embedding)
    labels_last = labels.copy()

    ClusteringLayer = clustering_layer_class()
    clustering = ClusteringLayer(args.clusters, name="clustering")(encoder.output)
    model = keras.Model(encoder.input, clustering, name="cpi_dec")
    model.compile(optimizer="adam", loss="kld")
    model.get_layer("clustering").set_weights([kmeans.cluster_centers_])

    p = None
    batch_start = 0
    rng = np.random.default_rng(args.seed)
    order = np.arange(n, dtype=np.int64)
    for iteration in range(args.max_iterations):
        if iteration % args.update_interval == 0:
            q = _predict_all(model, args.dataset, n, args.kind, args.batch_size)
            p = target_distribution(q)
            labels = q.argmax(axis=1)
            delta = np.mean(labels != labels_last)
            labels_last = labels.copy()
            print("iteration {}: delta_label={:.6g}".format(iteration, delta))
            if iteration > 0 and delta < args.tolerance:
                break

        if batch_start == 0:
            rng.shuffle(order)
        selected = order[batch_start: batch_start + args.batch_size]
        if len(selected) == 0:
            batch_start = 0
            continue
        x = load_batch(args.dataset, selected, args.kind)
        model.train_on_batch(x, p[selected])
        batch_start += args.batch_size
        if batch_start >= n:
            batch_start = 0

    output_model = Path(args.output_model)
    if output_model.suffix != ".keras":
        output_model = output_model.with_suffix(".keras")
    model.save(output_model)

    q = _predict_all(model, args.dataset, n, args.kind, args.batch_size)
    with h5py.File(args.output_encoding, "w") as handle:
        handle.create_dataset("probability", data=q, compression="gzip", shuffle=True)
        handle.create_dataset("label", data=q.argmax(axis=1).astype(np.int16))
        handle.create_dataset("embedding", data=embedding, compression="gzip", shuffle=True)


if __name__ == "__main__":
    main()
