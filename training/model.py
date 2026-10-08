import tensorflow as tf


def create_model(class_count):
    layers = tf.keras.layers
    model = tf.keras.Sequential(
        [
            layers.Input(shape=(64, 64, 3)),
            layers.Conv2D(32, 3, padding="same", use_bias=False),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.MaxPooling2D(2),
            layers.SeparableConv2D(64, 3, padding="same", use_bias=False),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.MaxPooling2D(2),
            layers.SeparableConv2D(128, 3, padding="same", use_bias=False),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.MaxPooling2D(2),
            layers.SeparableConv2D(256, 3, padding="same", use_bias=False),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.MaxPooling2D(2),
            layers.GlobalAveragePooling2D(),
            layers.Dense(1024, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(class_count, activation="softmax"),
        ]
    )
    return model
