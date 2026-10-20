import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tensorflow as tf

from model_registry import (
    file_hashes,
    labels_sha256,
    load_labels,
    sha256_file,
    write_json,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export a registered H5 model run directly to TFJS."
    )
    parser.add_argument(
        "run",
        type=Path,
        help="Run directory containing manifest.json, labels, and an H5 checkpoint",
    )
    parser.add_argument("--quantization-bytes", type=int, choices=(1, 2))
    return parser.parse_args()


def converter_path():
    candidate = Path(sys.executable).with_name(
        "tensorflowjs_converter.exe" if os.name == "nt" else "tensorflowjs_converter"
    )
    if candidate.exists():
        return os.fspath(candidate)
    executable = shutil.which("tensorflowjs_converter")
    if not executable:
        raise FileNotFoundError(
            "tensorflowjs_converter is not installed in the active environment"
        )
    return executable


def verify_unquantized_weights(model, model_json, directory):
    keras_weights = {
        weight.name.removesuffix(":0"): weight.numpy() for weight in model.weights
    }
    verified = 0
    for group in model_json.get("weightsManifest", []):
        data = b"".join(
            (directory / relative_path).read_bytes() for relative_path in group["paths"]
        )
        offset = 0
        for spec in group["weights"]:
            if spec["dtype"] != "float32" or "quantization" in spec:
                raise ValueError(
                    f"Unexpected encoding for unquantized weight {spec['name']}"
                )
            count = int(np.prod(spec["shape"], dtype=np.int64))
            byte_count = count * np.dtype("<f4").itemsize
            exported = np.frombuffer(
                data, dtype="<f4", count=count, offset=offset
            ).reshape(spec["shape"])
            expected = keras_weights.get(spec["name"])
            if expected is None or not np.array_equal(exported, expected):
                raise ValueError(f"TFJS weight differs from H5 source: {spec['name']}")
            offset += byte_count
            verified += 1
        if offset != len(data):
            raise ValueError("TFJS shard data contains unreferenced bytes")
    if verified != len(keras_weights):
        raise ValueError(
            f"Verified {verified} TFJS weights but H5 contains {len(keras_weights)}"
        )
    return verified


def main():
    args = parse_args()
    run_directory = args.run.resolve()
    manifest_path = run_directory / "manifest.json"
    with open(manifest_path, encoding="utf-8") as file:
        manifest = json.load(file)

    checkpoint = run_directory / manifest["checkpoint"]
    labels_path = run_directory / manifest["labels"]
    classes = load_labels(labels_path)
    if labels_sha256(classes) != manifest["labels_sha256"]:
        raise ValueError("Labels do not match the run manifest")
    if (
        manifest.get("checkpoint_sha256")
        and sha256_file(checkpoint) != manifest["checkpoint_sha256"]
    ):
        raise ValueError("Checkpoint does not match the run manifest")

    model = tf.keras.models.load_model(checkpoint, compile=False)
    if int(model.output_shape[-1]) != len(classes):
        raise ValueError(
            f"Model outputs {model.output_shape[-1]} values for {len(classes)} labels"
        )

    final_directory = run_directory / "tfjs"
    temporary_parent = Path(
        tempfile.mkdtemp(prefix="catchtwo-tfjs-", dir=run_directory)
    )
    temporary_output = temporary_parent / "tfjs"
    command = [
        converter_path(),
        "--input_format=keras",
        "--output_format=tfjs_layers_model",
    ]
    if args.quantization_bytes:
        command.append(f"--quantization_bytes={args.quantization_bytes}")
    command.extend([os.fspath(checkpoint), os.fspath(temporary_output)])

    try:
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"TFJS conversion failed:\n{result.stderr}")

        with open(temporary_output / "model.json", encoding="utf-8") as file:
            model_json = json.load(file)
        if model_json.get("format") != "layers-model":
            raise ValueError(
                f"Expected a TFJS layers-model, got {model_json.get('format')}"
            )
        layers = model_json["modelTopology"]["model_config"]["config"]["layers"]
        exported_outputs = int(layers[-1]["config"]["units"])
        if exported_outputs != len(classes):
            raise ValueError(
                f"TFJS model has {exported_outputs} outputs for {len(classes)} labels"
            )
        for layer in layers:
            if layer.get("class_name") == "SeparableConv2D":
                config = layer["config"]
                config.pop("kernel_initializer", None)
                config.pop("kernel_regularizer", None)
                config.pop("kernel_constraint", None)
        write_json(temporary_output / "model.json", model_json)
        verified_weights = None
        if not args.quantization_bytes:
            verified_weights = verify_unquantized_weights(
                model, model_json, temporary_output
            )

        write_json(
            temporary_output / "labels.json",
            {"schema_version": 1, "classes": classes},
        )
        if final_directory.exists():
            shutil.rmtree(final_directory)
        os.replace(temporary_output, final_directory)
    finally:
        shutil.rmtree(temporary_parent, ignore_errors=True)

    manifest["tfjs"] = {
        "directory": "tfjs",
        "format": "layers-model",
        "source_checkpoint_sha256": sha256_file(checkpoint),
        "quantization_bytes": args.quantization_bytes or 4,
        "exact_weight_parity": verified_weights == len(model.weights)
        if verified_weights is not None
        else None,
        "verified_weight_count": verified_weights,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "tensorflow_version": tf.__version__,
        "files": file_hashes(final_directory),
    }
    write_json(manifest_path, manifest)
    print(f"TFJS export completed: {final_directory}")


if __name__ == "__main__":
    main()
