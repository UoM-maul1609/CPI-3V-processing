"""Keras 3 model definitions used by the CPI ML workflows."""

from __future__ import annotations


def build_cnn_autoencoder(image_shape=(128, 128, 1), embedding_dim=64):
    import keras
    from keras import layers

    inputs = keras.Input(shape=image_shape, name="image")
    x = layers.Conv2D(32, 5, strides=2, activation="relu", padding="same")(inputs)
    x = layers.Conv2D(64, 5, strides=2, activation="relu", padding="same")(x)
    x = layers.Conv2D(128, 3, strides=2, activation="relu", padding="same")(x)
    x = layers.Conv2D(256, 3, strides=2, activation="relu", padding="same")(x)
    x = layers.Flatten()(x)
    embedding = layers.Dense(embedding_dim, activation="relu", name="embedding")(x)
    x = layers.Dense(8 * 8 * 256, activation="relu")(embedding)
    x = layers.Reshape((8, 8, 256))(x)
    x = layers.Conv2DTranspose(128, 3, strides=2, activation="relu", padding="same")(x)
    x = layers.Conv2DTranspose(64, 3, strides=2, activation="relu", padding="same")(x)
    x = layers.Conv2DTranspose(32, 5, strides=2, activation="relu", padding="same")(x)
    outputs = layers.Conv2DTranspose(1, 5, strides=2, padding="same", name="reconstruction")(x)
    return keras.Model(inputs, outputs, name="cpi_cnn_autoencoder")


def build_property_autoencoder(feature_dim=14, embedding_dim=10):
    import keras
    from keras import layers

    inputs = keras.Input(shape=(feature_dim,), name="properties")
    x = layers.Dense(500, activation="relu", kernel_initializer="glorot_uniform")(inputs)
    x = layers.Dense(500, activation="relu", kernel_initializer="glorot_uniform")(x)
    x = layers.Dense(2000, activation="relu", kernel_initializer="glorot_uniform")(x)
    embedding = layers.Dense(
        embedding_dim,
        activation="relu",
        kernel_initializer="glorot_uniform",
        name="embedding",
    )(x)
    x = layers.Dense(2000, activation="relu", kernel_initializer="glorot_uniform")(embedding)
    x = layers.Dense(500, activation="relu", kernel_initializer="glorot_uniform")(x)
    x = layers.Dense(500, activation="relu", kernel_initializer="glorot_uniform")(x)
    outputs = layers.Dense(feature_dim, activation="relu", name="reconstruction")(x)
    return keras.Model(inputs, outputs, name="cpi_property_autoencoder")


def encoder_from_autoencoder(model):
    import keras

    return keras.Model(model.input, model.get_layer("embedding").output, name=model.name + "_encoder")
