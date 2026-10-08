import argparse
import json
import os
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tensorflow as tf

from data_loader import ALIASES, load_split, make_dataset, oversample
from model_registry import (
    SCHEMA_VERSION,
    labels_sha256,
    load_labels,
    sha256_file,
    write_json,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PARENT_RUN = (
    SCRIPT_DIR / "models" / "runs" / "20260723T122145Z-sepconv-1279-real-ft"
)
DEFAULT_TRAIN_DIR = SCRIPT_DIR.parent / "datasets" / "catchtwo-dataset-100"
DEFAULT_VALIDATION_DIR = SCRIPT_DIR.parent / "datasets" / "catchtwo-dataset-validation"
DEFAULT_RUNS_DIR = SCRIPT_DIR / "models" / "runs"


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune the published model.")
    parser.add_argument("--parent-run", type=Path, default=DEFAULT_PARENT_RUN)
    parser.add_argument("--train-dir", type=Path, default=DEFAULT_TRAIN_DIR)
    parser.add_argument("--validation-dir", type=Path, default=DEFAULT_VALIDATION_DIR)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--run-id")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--target-images-per-class", type=int, default=100)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument(
        "--no-augmentation", dest="augmentation", action="store_false", default=True
    )
    parser.add_argument(
        "--no-oversampling", dest="oversampling", action="store_false", default=True
    )
    return parser.parse_args()


def configure_trainable_layers(model):
    for layer in model.layers:
        layer.trainable = False
    separable = [
        layer
        for layer in model.layers
        if isinstance(layer, tf.keras.layers.SeparableConv2D)
    ]
    dense = [
        layer for layer in model.layers if isinstance(layer, tf.keras.layers.Dense)
    ]
    if len(separable) < 2 or len(dense) < 2:
        raise ValueError(
            "Expected a separable convolutional model with two Dense layers"
        )
    for layer in separable[-2:] + dense[-2:]:
        layer.trainable = True
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True
    return [layer.name for layer in model.layers if layer.trainable]


def make_optimizer(learning_rate, steps):
    schedule = tf.keras.optimizers.schedules.CosineDecay(learning_rate, max(1, steps))
    return tf.keras.optimizers.Adam(learning_rate=schedule)


class BestCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, path, initial_loss, patience, min_delta):
        super().__init__()
        self.path = path
        self.best = float(initial_loss)
        self.best_epoch = 0
        self.patience = patience
        self.min_delta = min_delta
        self.wait = 0

    def on_epoch_end(self, epoch, logs=None):
        loss = (logs or {}).get("val_loss")
        if loss is not None and loss < self.best - self.min_delta:
            self.best = float(loss)
            self.best_epoch = epoch + 1
            self.wait = 0
            self.model.save(self.path)
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.stop_training = True


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    parent_run = args.parent_run.resolve()
    parent_manifest = json.loads(
        (parent_run / "manifest.json").read_text(encoding="utf-8")
    )
    classes = load_labels(parent_run / parent_manifest["labels"])
    checkpoint = parent_run / parent_manifest["checkpoint"]
    if sha256_file(checkpoint) != parent_manifest["checkpoint_sha256"]:
        raise ValueError("Parent checkpoint hash does not match its manifest")
    if labels_sha256(classes) != parent_manifest["labels_sha256"]:
        raise ValueError("Parent labels hash does not match its manifest")

    train_paths, train_labels, missing_train = load_split(
        args.train_dir.resolve(), classes
    )
    val_paths, val_labels, missing_val = load_split(
        args.validation_dir.resolve(), classes
    )

    model = tf.keras.models.load_model(checkpoint, compile=False)
    if tuple(model.input_shape[1:]) != (64, 64, 3) or int(
        model.output_shape[-1]
    ) != len(classes):
        raise ValueError(
            "Parent model input or output shape does not match this label set"
        )
    trainable = configure_trainable_layers(model)
    raw_train_count = len(train_paths)
    if args.oversampling:
        train_paths, train_labels = oversample(
            train_paths,
            train_labels,
            len(classes),
            args.target_images_per_class,
            args.seed,
        )

    loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=args.label_smoothing)
    train_data = make_dataset(
        train_paths,
        train_labels,
        len(classes),
        args.batch_size,
        True,
        args.seed,
        args.augmentation,
    )
    val_data = make_dataset(
        val_paths, val_labels, len(classes), args.batch_size, False, args.seed, False
    )
    steps = (
        max(1, (len(train_paths) + args.batch_size - 1) // args.batch_size)
        * args.epochs
    )
    model.compile(
        optimizer=make_optimizer(args.learning_rate, steps),
        loss=loss,
        metrics=[
            "accuracy",
            tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top5_accuracy"),
        ],
    )
    initial = model.evaluate(val_data, verbose=1, return_dict=True)

    run_id = args.run_id or datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ-sepconv-1279-ft"
    )
    run_dir = args.runs_dir.resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    output_checkpoint = run_dir / "best.h5"
    shutil.copy2(checkpoint, output_checkpoint)
    write_json(
        run_dir / "labels.json", {"schema_version": SCHEMA_VERSION, "classes": classes}
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "kind": "fine_tune",
        "status": "running",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "parent_run": parent_manifest["run_id"],
        "parent_checkpoint_sha256": parent_manifest["checkpoint_sha256"],
        "checkpoint": "best.h5",
        "labels": "labels.json",
        "labels_sha256": labels_sha256(classes),
        "class_count": len(classes),
        "train_directory": args.train_dir.name,
        "validation_directory": args.validation_dir.name,
        "train_image_count": raw_train_count,
        "effective_train_image_count": len(train_paths),
        "validation_image_count": len(val_paths),
        "missing_train_classes": missing_train,
        "missing_validation_classes": missing_val,
        "hyperparameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "label_smoothing": args.label_smoothing,
            "seed": args.seed,
            "oversampling": args.oversampling,
            "target_images_per_class": args.target_images_per_class
            if args.oversampling
            else None,
            "augmentation": args.augmentation,
            "trainable_layers": trainable,
        },
        "initial_validation": {key: float(value) for key, value in initial.items()},
        "tensorflow_version": tf.__version__,
    }
    write_json(run_dir / "manifest.json", manifest)

    tracker = BestCheckpoint(
        output_checkpoint, initial["loss"], args.patience, args.min_delta
    )
    history = model.fit(
        train_data, validation_data=val_data, epochs=args.epochs, callbacks=[tracker]
    )
    write_json(
        run_dir / "history.json",
        {key: [float(x) for x in values] for key, values in history.history.items()},
    )

    best_model = tf.keras.models.load_model(output_checkpoint, compile=False)
    best_model.compile(
        loss=loss,
        metrics=[
            "accuracy",
            tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top5_accuracy"),
        ],
    )
    final = best_model.evaluate(val_data, verbose=1, return_dict=True)
    manifest.update(
        {
            "status": "completed",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "checkpoint_sha256": sha256_file(output_checkpoint),
            "best_epoch": tracker.best_epoch,
            "best_validation": {key: float(value) for key, value in final.items()},
        }
    )
    write_json(run_dir / "manifest.json", manifest)
    print(f"Fine-tuned model saved to {run_dir}")


if __name__ == "__main__":
    main()
