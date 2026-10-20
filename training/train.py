import argparse
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
DEFAULT_LABELS = (
    SCRIPT_DIR
    / "models"
    / "runs"
    / "20260723T122145Z-sepconv-1279-real-ft"
    / "labels.json"
)
DEFAULT_TRAIN_DIR = SCRIPT_DIR.parent / "datasets" / "catchtwo-dataset-100"
DEFAULT_VALIDATION_DIR = SCRIPT_DIR.parent / "datasets" / "catchtwo-dataset-validation"
DEFAULT_RUNS_DIR = SCRIPT_DIR / "models" / "runs"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a model from random initialization."
    )
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--train-dir", type=Path, default=DEFAULT_TRAIN_DIR)
    parser.add_argument("--validation-dir", type=Path, default=DEFAULT_VALIDATION_DIR)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--run-id")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--target-images-per-class", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument(
        "--no-augmentation", dest="augmentation", action="store_false", default=True
    )
    parser.add_argument(
        "--no-oversampling", dest="oversampling", action="store_false", default=True
    )
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    classes = load_labels(args.labels)
    train_paths, train_labels, missing_train = load_split(
        args.train_dir.resolve(), classes
    )
    val_paths, val_labels, missing_val = load_split(
        args.validation_dir.resolve(), classes
    )
    raw_train_count = len(train_paths)
    if args.oversampling:
        train_paths, train_labels = oversample(
            train_paths,
            train_labels,
            len(classes),
            args.target_images_per_class,
            args.seed,
        )

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
    model = create_model(len(classes))
    loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=args.label_smoothing)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss=loss,
        metrics=[
            "accuracy",
            tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top5_accuracy"),
        ],
    )

    run_id = args.run_id or datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ-sepconv-train"
    )
    run_dir = args.runs_dir.resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    checkpoint = run_dir / "best.h5"
    write_json(
        run_dir / "labels.json", {"schema_version": SCHEMA_VERSION, "classes": classes}
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "kind": "from_scratch_training",
        "status": "running",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "parent_run": None,
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
            "target_images_per_class": args.target_images_per_class
            if args.oversampling
            else None,
            "oversampling": args.oversampling,
            "augmentation": args.augmentation,
            "seed": args.seed,
        },
        "tensorflow_version": tf.__version__,
    }
    write_json(run_dir / "manifest.json", manifest)
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=args.epochs,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                checkpoint, monitor="val_loss", save_best_only=True, mode="min"
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.3, patience=4, min_lr=1e-6, mode="min"
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=15, restore_best_weights=True
            ),
        ],
    )
    write_json(
        run_dir / "history.json",
        {key: [float(x) for x in values] for key, values in history.history.items()},
    )
    best_model = tf.keras.models.load_model(checkpoint, compile=False)
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
            "checkpoint_sha256": sha256_file(checkpoint),
            "best_validation": {key: float(value) for key, value in final.items()},
        }
    )
    write_json(run_dir / "manifest.json", manifest)
    print(f"Trained model saved to {run_dir}")


if __name__ == "__main__":
    main()
