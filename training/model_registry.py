import hashlib
import json
import os
from pathlib import Path

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
SCHEMA_VERSION = 1


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def labels_sha256(classes):
    data = json.dumps(classes, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def load_labels(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    classes = value.get("classes") if isinstance(value, dict) else value
    if (
        not isinstance(classes, list)
        or not classes
        or not all(isinstance(name, str) for name in classes)
    ):
        raise ValueError(f"Invalid ordered label file: {path}")
    if len(classes) != len(set(classes)):
        raise ValueError(f"Duplicate labels in: {path}")
    return classes


def class_directories(root):
    root = os.path.abspath(root)
    result = {}
    for name in os.listdir(root):
        path = os.path.join(root, name)
        # Preserve class folder names that end in a dot or space on Windows.
        if os.name == "nt" and name.endswith((".", " ")):
            path = "\\\\?\\" + path
        if os.path.isdir(path):
            result[name] = path
    return result


def collect_labeled_images(root, classes, aliases=None):
    aliases = aliases or {}
    indices = {name: index for index, name in enumerate(classes)}
    directories = class_directories(root)
    unknown = sorted(
        name for name in directories if aliases.get(name, name) not in indices
    )
    if unknown:
        raise ValueError(f"Unknown classes in {root}: {unknown}")

    paths, labels = [], []
    present = set()
    for source_name, directory in sorted(directories.items()):
        class_name = aliases.get(source_name, source_name)
        for filename in sorted(os.listdir(directory)):
            if filename.lower().endswith(IMAGE_EXTENSIONS):
                path = os.path.join(directory, filename)
                if os.path.isfile(path):
                    paths.append(path)
                    labels.append(indices[class_name])
                    present.add(class_name)

    missing = [name for name in classes if name not in present]
    return paths, labels, missing


def file_hashes(root):
    return {
        path.name: sha256_file(path)
        for path in sorted(Path(root).iterdir())
        if path.is_file()
    }
