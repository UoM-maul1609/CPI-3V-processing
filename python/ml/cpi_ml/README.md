# CPI ML (modern path)

This package is the maintainable replacement for the collection of one-off Keras 2 scripts in
`python/ml/cnn`, `python/ml/sae`, and `python/ml/saeMNIST`.

The old scripts are retained for reproducing historical experiments/models, but new work should use
this package. The modern path uses public Keras 3 APIs, named `embedding` layers, `.keras` whole-model
files, reproducible splits, and HDF5 batch loading. Keras 3 recommends the native `.keras` format;
legacy JSON + weights files can still be migrated separately.

## Build a dataset

```bash
python -m cpi_ml.dataset postProcessed.h5 /data/C296 /data/C299 --focus 5 --min-length 50
```

The builder preserves the existing `postProcessImages05` scientific preprocessing, but each source
MAT file is handled by a disposable subprocess and appended to a chunked/compressed HDF5 file. It also
stores `source_file` and `roi_index`, which makes provenance and grouped validation possible.

## Train an image autoencoder

```bash
python -m cpi_ml.train_autoencoder postProcessed.h5 cnn.keras --kind image --epochs 100
```

By default the validation split is grouped by source file rather than randomly mixing particles from
the same acquisition file into train and validation sets.

## Train a property autoencoder

```bash
python -m cpi_ml.train_autoencoder postProcessed.h5 properties.keras --kind properties
```

The 14-property input is the historical 12 normalized diameters + roundness + length/width ratio.

## Encode / cluster

```bash
python -m cpi_ml.encode postProcessed.h5 cnn.keras encoding.h5 --kind image
python -m cpi_ml.cluster postProcessed.h5 cnn.keras dec.keras clusters.h5 --clusters 9
```

## Legacy warning

Several historical scripts used `if ~auxLoad:`. In Python, `~False` and `~True` are integers (`-1`
and `-2`), both truthy, so those conditions were always taken. They are fixed in this repository, but
old results produced with those scripts may have used newly generated train/test indices even when an
auxiliary split was intended to be loaded.
