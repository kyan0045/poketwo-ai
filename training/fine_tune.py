import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tensorflow as tf

from data_loader import load_split, make_dataset, oversample
from model import create_model
from model_registry import (
    SCHEMA_VERSION,
    labels_sha256,
    load_labels,
    sha256_file,
    write_json,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PARENT_RUN = (
    SCRIPT_DIR / "models" / "runs" / "20261102T141243Z-sepconv-accuracy-ft"
)
DEFAULT_TRAIN_DIR = SCRIPT_DIR.parent / "datasets" / "catchtwo-dataset-1303"
DEFAULT_VALIDATION_DIR = (
    SCRIPT_DIR.parent / "datasets" / "catchtwo-dataset-validation-1303"
)
DEFAULT_RUNS_DIR = SCRIPT_DIR / "models" / "runs"
TRUNK_LAYER_COUNT = 17


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune the compact GAP -> Dense model. The convolutional trunk "
            "is warm-started from the published checkpoint and every layer is "
            "trained end to end."
        )
    )
    parser.add_argument("--parent-run", type=Path, default=DEFAULT_PARENT_RUN)
    parser.add_argument("--train-dir", type=Path, default=DEFAULT_TRAIN_DIR)
    parser.add_argument("--validation-dir", type=Path, default=DEFAULT_VALIDATION_DIR)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--run-id")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--head-epochs",
        type=int,
        default=2,
        help="Epochs that train only the new Dense head before unfreezing.",
    )
    parser.add_argument("--head-learning-rate", type=float, default=1e-3)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--target-images-per-class", type=int, default=100)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--lr-factor", type=float, default=0.3)
    parser.add_argument("--lr-patience", type=int, default=2)
    parser.add_argument("--min-learning-rate", type=float, default=1e-6)
    parser.add_argument(
        "--augmentation",
        dest="augmentation",
        action="store_true",
        default=True,
        help="Enable training-time image augmentation (on by default).",
    )
    parser.add_argument(
        "--no-augmentation",
        dest="augmentation",
        action="store_false",
        help="Disable training-time image augmentation.",
    )
    parser.add_argument(
        "--oversampling",
        dest="oversampling",
        action="store_true",
        default=False,
        help="Oversample rare classes (off by default; v2 is balanced).",
    )
    return parser.parse_args()


def warm_start_trunk(model, parent_checkpoint, class_count):
    parent = tf.keras.models.load_model(parent_checkpoint, compile=False)
    if tuple(parent.input_shape[1:]) != (64, 64, 3):
        raise ValueError("Parent model input shape is not (64, 64, 3)")
    if int(parent.output_shape[-1]) != class_count:
        raise ValueError(
            f"Parent model has {parent.output_shape[-1]} outputs for "
            f"{class_count} labels"
        )

    for index, (compact_layer, parent_layer) in enumerate(
        zip(model.layers[:TRUNK_LAYER_COUNT], parent.layers[:TRUNK_LAYER_COUNT])
    ):
        if compact_layer.name != parent_layer.name:
            raise ValueError(
                f"Trunk layer {index} mismatch: {compact_layer.name} vs "
                f"{parent_layer.name}"
            )
        compact_weights = compact_layer.get_weights()
        parent_weights = parent_layer.get_weights()
        if [w.shape for w in compact_weights] != [w.shape for w in parent_weights]:
            raise ValueError(
                f"Trunk layer {index} ({compact_layer.name}) weight shapes differ"
            )
        compact_layer.set_weights(parent_weights)

    return [layer.name for layer in model.layers if layer.trainable]


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
    raw_train_count = len(train_paths)

    model = create_model(len(classes))
    warm_start_trunk(model, checkpoint, len(classes))
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

    def make_metrics():
        return [
            "accuracy",
            tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top5_accuracy"),
        ]

    if args.head_epochs > 0:
        for layer in model.layers[:TRUNK_LAYER_COUNT]:
            layer.trainable = False
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=args.head_learning_rate),
            loss=loss,
            metrics=make_metrics(),
        )
        model.fit(
            train_data,
            validation_data=val_data,
            epochs=args.head_epochs,
            verbose=1,
        )

    for layer in model.layers:
        layer.trainable = True
    trainable = [layer.name for layer in model.layers if layer.trainable]
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss=loss,
        metrics=make_metrics(),
    )
    initial = model.evaluate(val_data, verbose=1, return_dict=True)

    run_id = args.run_id or datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ-sepconv-1303-compact-ft"
    )
    run_dir = args.runs_dir.resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    output_checkpoint = run_dir / "best.h5"
    write_json(
        run_dir / "labels.json", {"schema_version": SCHEMA_VERSION, "classes": classes}
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "kind": "compact_fine_tune",
        "status": "running",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "architecture": "compact-gap-dense",
        "parameter_count": int(model.count_params()),
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
            "head_epochs": args.head_epochs,
            "head_learning_rate": args.head_learning_rate,
            "learning_rate_schedule": {
                "type": "ReduceLROnPlateau",
                "factor": args.lr_factor,
                "patience": args.lr_patience,
                "min_learning_rate": args.min_learning_rate,
            },
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
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=args.lr_factor,
        patience=args.lr_patience,
        min_lr=args.min_learning_rate,
        mode="min",
    )
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=args.epochs,
        callbacks=[tracker, lr_scheduler],
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
    print(f"Fine-tuned compact model saved to {run_dir}")


if __name__ == "__main__":
    main()
