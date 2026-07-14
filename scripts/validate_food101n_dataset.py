#!/usr/bin/env python3
"""Validate a local Food-101N/Food-101 layout before Mammoth training."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PATH_KEYS = ("file_name", "filepath", "file_path", "path", "image", "img", "sample_key")
CLASS_KEYS = ("klass", "class", "class_name", "category", "category_name", "label_name")
LABEL_KEYS = ("label", "target", "class_idx", "class_id", "category_id")
VERIFICATION_KEYS = ("verification_label", "verified", "is_verified")
TASK_SPLIT = [20, 20, 20, 20, 21]
EXPECTED_TRAIN_RECORDS = 52867
EXPECTED_TEST_RECORDS = 4741


def clean_value(value: Any) -> Optional[str]:
    if value is None:
        return None
    value = str(value).strip()
    if value == "" or value.lower() in {"none", "nan", "null"}:
        return None
    return value


def is_int_like(value: Any) -> bool:
    value = clean_value(value)
    if value is None:
        return False
    try:
        int(value)
    except ValueError:
        return False
    return True


def first_value(row: Dict[str, Any], keys: Sequence[str]) -> Optional[str]:
    lowered = {str(k).strip().lower(): v for k, v in row.items()}
    for key in keys:
        value = clean_value(lowered.get(key))
        if value is not None:
            return value
    return None


def normalise_dict_record(row: Dict[str, Any]) -> Dict[str, Any]:
    record = dict(row)
    out: Dict[str, Any] = {}
    for out_key, keys in (
        ("file_name", PATH_KEYS),
        ("klass", CLASS_KEYS),
        ("label", LABEL_KEYS),
        ("verification_label", VERIFICATION_KEYS),
    ):
        value = first_value(record, keys)
        if value is not None:
            out[out_key] = value
    return out


def normalise_sequence_record(parts: Sequence[str]) -> Dict[str, Any]:
    record: Dict[str, Any] = {"file_name": parts[0]}
    if len(parts) > 1:
        if is_int_like(parts[1]):
            record["label"] = parts[1]
        else:
            record["klass"] = parts[1]
    if len(parts) > 2:
        if "klass" not in record and not is_int_like(parts[2]):
            record["klass"] = parts[2]
        elif is_int_like(parts[2]):
            record["verification_label"] = parts[2]
    return record


def read_json_records(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, list):
        return [
            normalise_dict_record(item) if isinstance(item, dict) else {"file_name": str(item)}
            for item in raw
        ]

    if not isinstance(raw, dict):
        raise ValueError(f"Unsupported JSON metadata format in {path}")

    for key in ("annotations", "samples", "data", "images", "items"):
        value = raw.get(key)
        if isinstance(value, list):
            return [
                normalise_dict_record(item) if isinstance(item, dict) else {"file_name": str(item)}
                for item in value
            ]

    records = []
    for klass, rel_paths in raw.items():
        if not isinstance(rel_paths, list):
            continue
        for rel_path in rel_paths:
            rel_path = str(rel_path)
            if not rel_path.startswith(f"{klass}/"):
                rel_path = f"{klass}/{rel_path}"
            records.append({"file_name": rel_path, "klass": str(klass)})
    if records:
        return records

    raise ValueError(f"Unsupported JSON metadata format in {path}")


def read_text_records(path: Path) -> List[Dict[str, Any]]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    lines = [line for line in lines if line and not line.startswith("#")]
    if not lines:
        return []

    delimiter = "\t" if "\t" in lines[0] else "," if "," in lines[0] else None
    first = lines[0].lower()
    header_tokens = set(first.replace("\t", ",").replace(" ", ",").split(","))
    known_keys = set(PATH_KEYS + CLASS_KEYS + LABEL_KEYS + VERIFICATION_KEYS)

    if delimiter is not None and header_tokens.intersection(known_keys):
        reader = csv.DictReader(lines, delimiter=delimiter)
        return [normalise_dict_record(row) for row in reader]

    records = []
    for line in lines:
        parts = line.split(delimiter) if delimiter is not None else line.split()
        parts = [clean_value(part) for part in parts]
        parts = [part for part in parts if part is not None]
        if parts:
            records.append(normalise_sequence_record(parts))
    return records


def read_metadata_records(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".json":
        return read_json_records(path)
    return read_text_records(path)


def load_classes_file(path: Optional[Path]) -> Optional[List[str]]:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"Classes file not found: {path}")

    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, dict) and "classes" in raw:
            raw = raw["classes"]
        if isinstance(raw, dict):
            return [str(k) for k in sorted(raw.keys())]
        if isinstance(raw, list):
            return [str(item) for item in raw]
        raise ValueError(f"Unsupported classes JSON format: {path}")

    classes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        classes.append(parts[1] if len(parts) > 1 and is_int_like(parts[0]) else parts[0])
    return classes


def record_class_name(record: Dict[str, Any]) -> Optional[str]:
    klass = clean_value(record.get("klass"))
    if klass is not None:
        return klass
    path = clean_value(record.get("file_name"))
    if path is None:
        return None
    parts = Path(path).parts
    return parts[0] if len(parts) > 1 else None


def record_label(record: Dict[str, Any]) -> Optional[int]:
    label = clean_value(record.get("label"))
    return int(label) if is_int_like(label) else None


def infer_classes(records: Sequence[Dict[str, Any]]) -> List[str]:
    names = set()
    label_to_name: Dict[int, str] = {}
    label_values = []

    for record in records:
        klass = record_class_name(record)
        label = record_label(record)
        if klass is not None:
            names.add(klass)
        if label is not None:
            label_values.append(label)
        if klass is not None and label is not None:
            old_name = label_to_name.get(label)
            if old_name is not None and old_name != klass:
                raise ValueError(f"Conflicting class names for label {label}: {old_name} vs {klass}")
            label_to_name[label] = klass

    if label_to_name and len(label_to_name) == len(names):
        labels = sorted(label_to_name)
        if labels == list(range(labels[-1] + 1)):
            return [label_to_name[i] for i in labels]

    if names:
        return sorted(names)

    if label_values:
        max_label = max(label_values)
        return [f"food_class_{i:03d}" for i in range(max_label + 1)]

    raise ValueError("Could not infer class names from metadata.")


def scan_image_records(images_dir: Path) -> List[Dict[str, Any]]:
    if not images_dir.exists():
        return []
    records = []
    for class_dir in sorted([p for p in images_dir.iterdir() if p.is_dir()]):
        for image_path in sorted(class_dir.rglob("*")):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_SUFFIXES:
                records.append({
                    "file_name": str(image_path.relative_to(images_dir)),
                    "klass": class_dir.name,
                })
    return records


def first_existing(paths: Sequence[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists():
            return path
    return None


def default_metadata_path(root: Path, train: bool) -> Optional[Path]:
    names = (
        ("meta/train.json", "meta/train.txt", "meta/train.tsv", "meta/train.csv",
         "train.json", "train.txt", "train.tsv", "train.csv")
        if train else
        ("meta/test.json", "meta/test.txt", "meta/test.tsv", "meta/test.csv",
         "test.json", "test.txt", "test.tsv", "test.csv")
    )
    return first_existing([root / name for name in names])


def default_images_dir(root: Path, train: bool) -> Path:
    names = (
        ("train", "Food-101N_release/train", "images", ".")
        if train else
        ("test", "val", "images", "Food-101/images", "food-101/images", ".")
    )
    existing = first_existing([root / name for name in names])
    return existing if existing is not None else root / ("train" if train else "test")


def suffix_candidates(path: Path) -> Iterable[Path]:
    if path.suffix:
        yield path
        return
    for suffix in sorted(IMAGE_SUFFIXES):
        yield path.with_suffix(suffix)


def resolve_image_path(path_value: str, images_dir: Path) -> Path:
    path = Path(path_value)
    base = path if path.is_absolute() else images_dir / path
    for candidate in suffix_candidates(base):
        if candidate.exists():
            return candidate
    return next(iter(suffix_candidates(base)))


def resolve_optional_path(value: str, root: Path) -> Optional[Path]:
    if value == "":
        return None
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    root_candidate = root / path
    repo_candidate = Path.cwd() / path
    if root_candidate.exists() or not repo_candidate.exists():
        return root_candidate
    return repo_candidate


def load_records(split_name: str, root: Path, metadata_path: Optional[Path], images_dir: Path) -> List[Dict[str, Any]]:
    if metadata_path is not None:
        if not metadata_path.exists():
            raise FileNotFoundError(f"{split_name} metadata file not found: {metadata_path}")
        return read_metadata_records(metadata_path)

    records = scan_image_records(images_dir)
    if not records:
        raise FileNotFoundError(
            f"No {split_name} records found. Expected metadata under {root}/meta "
            f"or images arranged as {images_dir}/<class_name>/<image>."
        )
    return records


def validate_split(
    split_name: str,
    root: Path,
    metadata_path: Optional[Path],
    images_dir: Path,
    class_to_idx: Optional[Dict[str, int]],
    classes: Optional[List[str]],
    strict: bool,
    skip_image_path_check: bool,
    max_missing_report: int,
) -> Tuple[Dict[str, int], Dict[str, int]]:
    records = load_records(split_name, root, metadata_path, images_dir)
    if classes is None and class_to_idx is None:
        classes = infer_classes(records)
    if class_to_idx is None:
        assert classes is not None
        class_to_idx = {klass: idx for idx, klass in enumerate(classes)}

    labels = []
    missing = []
    verified_count = 0
    for record in records:
        path_value = clean_value(record.get("file_name"))
        if path_value is None:
            raise ValueError(f"{split_name} record has no image path: {record}")

        if not skip_image_path_check:
            image_path = resolve_image_path(path_value, images_dir)
            if not image_path.exists():
                missing.append(image_path)

        label = record_label(record)
        klass = record_class_name(record)
        if label is None:
            if klass is None or klass not in class_to_idx:
                raise ValueError(f"Cannot resolve {split_name} label for record: {record}")
            label = class_to_idx[klass]
        elif label < 0 or label >= len(class_to_idx):
            raise ValueError(f"{split_name} label {label} out of range for {len(class_to_idx)} classes.")

        if klass is not None and klass in class_to_idx and class_to_idx[klass] != label:
            raise ValueError(
                f"{split_name} label/class mismatch for {path_value}: "
                f"label={label}, class={klass}, class_to_idx={class_to_idx[klass]}"
            )
        if clean_value(record.get("verification_label")) is not None:
            verified_count += 1
        labels.append(label)

    if strict and missing:
        examples = "\n".join(str(path) for path in missing[:max_missing_report])
        raise FileNotFoundError(
            f"{split_name} has {len(missing)} missing image paths. First missing paths:\n{examples}"
        )

    unique_labels = sorted(set(labels))
    print(f"[{split_name}] records: {len(records)}")
    print(f"[{split_name}] metadata: {metadata_path or '<scanned image folders>'}")
    print(f"[{split_name}] images_dir: {images_dir}")
    print(f"[{split_name}] classes: {len(class_to_idx)}")
    print(f"[{split_name}] label range: {unique_labels[0]}..{unique_labels[-1]}")
    print(f"[{split_name}] verified flag records: {verified_count}")
    if skip_image_path_check:
        print(f"[{split_name}] missing image paths: <skipped>")
    else:
        print(f"[{split_name}] missing image paths: {len(missing)}")
    if missing and not strict:
        for path in missing[:max_missing_report]:
            print(f"[{split_name}] missing example: {path}")

    return class_to_idx, {
        "records": len(records),
        "classes": len(class_to_idx),
        "missing": len(missing),
        "min_label": unique_labels[0],
        "max_label": unique_labels[-1],
    }


def check_mammoth_import(args: argparse.Namespace) -> None:
    try:
        from argparse import Namespace

        from datasets.seq_food101n import SequentialFood101N
    except Exception as exc:  # pragma: no cover - depends on project env
        raise RuntimeError(f"Could not import Mammoth Food101N dataset: {exc}") from exc

    ds_args = Namespace(
        dataset="seq-food101n",
        noise_rate=0,
        joint=0,
        custom_task_order=None,
        custom_class_order=None,
        permute_classes=False,
        class_order=None,
        label_perc=1,
        label_perc_by_class=1,
        seed=0,
        validation=None,
        validation_mode="current",
        batch_size=2,
        drop_last=False,
        num_workers=0,
    )
    dataset = SequentialFood101N(
        ds_args,
        food101n_root=str(args.root),
        food101n_train_list=str(args.train_list or ""),
        food101n_test_list=str(args.test_list or ""),
        food101n_images_dir=str(args.images_dir or ""),
        food101n_train_images_dir=str(args.train_images_dir or ""),
        food101n_test_images_dir=str(args.test_images_dir or ""),
        food101n_classes_file=str(args.classes_file or ""),
        food101n_strict=int(args.strict),
    )
    train_loader, test_loader = dataset.get_data_loaders()
    print("[mammoth] SequentialFood101N import and loader construction: ok")
    if args.check_batch:
        next(iter(train_loader))
        next(iter(test_loader))
        print("[mammoth] one train/test batch read: ok")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/Food-101N", help="Food-101N root path.")
    parser.add_argument("--train-list", default="", help="Train metadata path. Default probes root/meta/train.*.")
    parser.add_argument("--test-list", default="", help="Test metadata path. Default probes root/meta/test.*.")
    parser.add_argument("--images-dir", default="", help="Shared image directory override.")
    parser.add_argument("--train-images-dir", default="", help="Train image directory override.")
    parser.add_argument("--test-images-dir", default="", help="Test image directory override.")
    parser.add_argument("--classes-file", default="", help="Optional 101-class names file.")
    parser.add_argument("--expected-classes", type=int, default=101)
    parser.add_argument(
        "--expected-train-records",
        type=int,
        default=EXPECTED_TRAIN_RECORDS,
        help="Expected Food-101N train records for the AER/NTD-style split. Set 0 to disable.",
    )
    parser.add_argument(
        "--expected-test-records",
        type=int,
        default=EXPECTED_TEST_RECORDS,
        help="Expected Food-101N test records for the AER/NTD-style split. Set 0 to disable.",
    )
    parser.add_argument("--strict", action="store_true", help="Fail if any referenced image path is missing.")
    parser.add_argument("--skip-image-path-check", action="store_true", help="Only validate metadata and labels.")
    parser.add_argument("--max-missing-report", type=int, default=10)
    parser.add_argument("--check-import", action="store_true", help="Also instantiate datasets.seq_food101n.")
    parser.add_argument("--check-batch", action="store_true", help="With --check-import, read one train/test batch.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.root = Path(args.root).expanduser()
    if not args.root.is_absolute():
        args.root = Path.cwd() / args.root

    args.train_list = resolve_optional_path(args.train_list, args.root)
    args.test_list = resolve_optional_path(args.test_list, args.root)
    args.images_dir = resolve_optional_path(args.images_dir, args.root)
    args.train_images_dir = resolve_optional_path(args.train_images_dir, args.root)
    args.test_images_dir = resolve_optional_path(args.test_images_dir, args.root)
    args.classes_file = resolve_optional_path(args.classes_file, args.root)

    train_metadata = args.train_list or default_metadata_path(args.root, train=True)
    test_metadata = args.test_list or default_metadata_path(args.root, train=False)
    train_images_dir = args.train_images_dir or args.images_dir or default_images_dir(args.root, train=True)
    test_images_dir = args.test_images_dir or args.images_dir or default_images_dir(args.root, train=False)
    classes = load_classes_file(args.classes_file)

    print(f"[food101n] root: {args.root}")
    print(f"[food101n] task split: {TASK_SPLIT} (sum={sum(TASK_SPLIT)})")
    if args.expected_train_records > 0 or args.expected_test_records > 0:
        print(
            "[food101n] expected split records: "
            f"train={args.expected_train_records or '<disabled>'}, "
            f"test={args.expected_test_records or '<disabled>'}"
        )

    class_to_idx, train_stats = validate_split(
        "train",
        args.root,
        train_metadata,
        train_images_dir,
        class_to_idx=None,
        classes=classes,
        strict=args.strict,
        skip_image_path_check=args.skip_image_path_check,
        max_missing_report=args.max_missing_report,
    )
    _, test_stats = validate_split(
        "test",
        args.root,
        test_metadata,
        test_images_dir,
        class_to_idx=class_to_idx,
        classes=classes,
        strict=args.strict,
        skip_image_path_check=args.skip_image_path_check,
        max_missing_report=args.max_missing_report,
    )

    errors = []
    for split_name, stats in (("train", train_stats), ("test", test_stats)):
        if stats["classes"] != args.expected_classes:
            errors.append(f"{split_name} has {stats['classes']} classes, expected {args.expected_classes}")
        if stats["min_label"] != 0 or stats["max_label"] != args.expected_classes - 1:
            errors.append(
                f"{split_name} label range is {stats['min_label']}..{stats['max_label']}, "
                f"expected 0..{args.expected_classes - 1}"
            )

    if args.expected_train_records > 0 and train_stats["records"] != args.expected_train_records:
        errors.append(
            f"train has {train_stats['records']} records, expected {args.expected_train_records} "
            "for the selected AER/NTD-style Food101N split"
        )
    if args.expected_test_records > 0 and test_stats["records"] != args.expected_test_records:
        errors.append(
            f"test has {test_stats['records']} records, expected {args.expected_test_records} "
            "for the selected AER/NTD-style Food101N split"
        )

    if sum(TASK_SPLIT) != args.expected_classes:
        errors.append(f"Task split {TASK_SPLIT} does not sum to {args.expected_classes}")

    if args.check_batch and not args.check_import:
        errors.append("--check-batch requires --check-import")

    if errors:
        print("[food101n] validation failed:")
        for error in errors:
            print(f"  - {error}")
        return 1

    if args.check_import:
        check_mammoth_import(args)

    print("[food101n] validation passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
