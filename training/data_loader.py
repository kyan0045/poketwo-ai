import random

import tensorflow as tf

from model_registry import collect_labeled_images


IMAGE_SIZE = 64
ALIASES = {"Mime Jr.": "Mime Jr", "Mime Jr_": "Mime Jr", "MissingNo.": "MissingNo"}


def load_split(directory, classes):
    paths, labels, missing = collect_labeled_images(directory, classes, ALIASES)
    if not paths:
        raise ValueError(f"No supported images found in {directory}")
    return paths, labels, missing


def augment_image(image):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.12)
    image = tf.image.random_contrast(image, lower=0.85, upper=1.15)
    image = tf.image.random_saturation(image, 0.85, 1.15)
    image = tf.image.random_hue(image, max_delta=0.02)
    image = tf.clip_by_value(image, 0.0, 1.0)
    image = tf.cast(image * 255.0, tf.uint8)
    image = tf.image.random_jpeg_quality(image, 60, 95)
    image = tf.cast(image, tf.float32) / 255.0
    zoom = tf.random.uniform([], 1.0, 1.15)
    size = tf.cast(tf.round(IMAGE_SIZE * zoom), tf.int32)
    image = tf.image.resize(image, [size, size])
    offset = (size - IMAGE_SIZE) // 2
    return tf.image.crop_to_bounding_box(image, offset, offset, IMAGE_SIZE, IMAGE_SIZE)


def oversample(paths, labels, class_count, target, seed):
    grouped = {index: [] for index in range(class_count)}
    for path, label in zip(paths, labels):
        grouped[int(label)].append(path)
    rng = random.Random(seed)
    samples = []
    for label, items in grouped.items():
        if items and len(items) < target:
            items = (items * ((target // len(items)) + 1))[:target]
        samples.extend((path, label) for path in items)
    rng.shuffle(samples)
    return [item[0] for item in samples], [item[1] for item in samples]


def make_dataset(paths, labels, class_count, batch_size, training, seed, augment):
    def preprocess(path, label):
        image = tf.io.read_file(path)
        image = tf.io.decode_image(image, channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE]) / 255.0
        if training and augment:
            image = augment_image(image)
        return image, tf.one_hot(label, class_count)

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        dataset = dataset.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)
    return (
        dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
