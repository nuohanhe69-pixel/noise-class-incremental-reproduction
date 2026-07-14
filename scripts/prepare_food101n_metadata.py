#!/usr/bin/env python3
"""Prepare canonical Food-101N metadata for the selected 52,867/4,741 split."""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from validate_food101n_dataset import (
    EXPECTED_TEST_RECORDS,
    EXPECTED_TRAIN_RECORDS,
    clean_value,
    infer_classes,
    load_classes_file,
    read_metadata_records,
    record_class_name,
    record_label,
    resolve_image_path,
    scan_image_records,
)


FIELDNAMES = ("file_name", "klass", "label", "verification_label")


def resolve_path(value: str, base: Optional[Path] = None) -> Optional[Path]:
    if value == "":
        return None
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (base or Path.cwd()) / path


def load_records(split_name: str, metadata_path: Optional[Path], images_dir: Path) -> List[Dict[str, str]]:
    if metadata_path is not None:
        if not metadata_path.exists():
            raise FileNotFoundError(f"{split_name} metadata not found: {metadata_path}")
        return read_metadata_records(metadata_path)

    records = scan_image_records(images_dir)
    if not records:
        raise FileNotFoundError(
            f"No {split_name} records found. Provide --{split_name}-list or arrange images as "
            f"{images_dir}/<class_name>/<image>."
        )
    return records


def load_task_records(
    task_dir: Path,
    split_name: str,
    dataset: str,
    exp: str,
    rand: int,
    classes_per_task: int,
    n_tasks: int,
) -> List[Dict[str, str]]:
    records: List[Dict[str, str]] = []
    for task_idx in range(n_tasks):
        if split_name == "train":
            name = f"{dataset}_train_{exp}_rand{rand}_cls{classes_per_task}_task{task_idx}.json"
        else:
            name = f"{dataset}_test_rand{rand}_cls{classes_per_task}_task{task_idx}.json"
        path = task_dir / name
        if not path.exists():
            raise FileNotFoundError(f"{split_name} task metadata not found: {path}")
        records.extend(read_metadata_records(path))
    return records


def relative_source_path(path_value: str, source_path: Path, images_dir: Path, klass: str) -> Path:
    try:
        return source_path.relative_to(images_dir)
    except ValueError:
        raw_path = Path(path_value)
        if not raw_path.is_absolute() and len(raw_path.parts) > 1:
            return raw_path
        return Path(klass) / source_path.name


def materialize_image(source_path: Path, target_path: Path, copy_images: bool, link_images: bool) -> None:
    if not copy_images and not link_images:
        return
    if target_path.exists() or target_path.is_symlink():
        return
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if link_images:
        target_path.symlink_to(source_path)
    else:
        shutil.copy2(source_path, target_path)


def normalize_records(
    split_name: str,
    records: Sequence[Dict[str, str]],
    images_dir: Path,
    classes: Sequence[str],
    strict: bool,
    output_images_dir: Path,
    copy_images: bool,
    link_images: bool,
) -> Tuple[List[Dict[str, str]], int]:
    class_to_idx = {klass: idx for idx, klass in enumerate(classes)}
    idx_to_class = {idx: klass for klass, idx in class_to_idx.items()}
    rows: List[Dict[str, str]] = []
    missing = 0

    for record in records:
        path_value = clean_value(record.get("file_name"))
        if path_value is None:
            raise ValueError(f"{split_name} record has no file_name: {record}")

        source_path = resolve_image_path(path_value, images_dir)
        if not source_path.exists():
            missing += 1
            if strict:
                raise FileNotFoundError(f"{split_name} image not found: {source_path}")
            if copy_images or link_images:
                raise FileNotFoundError(
                    f"{split_name} image not found and cannot be materialized: {source_path}. "
                    "Download images first or omit --copy-images/--link-images."
                )

        label = record_label(record)
        klass = record_class_name(record)

        if label is None:
            if klass is None or klass not in class_to_idx:
                raise ValueError(f"Cannot resolve {split_name} label for record: {record}")
            label = class_to_idx[klass]
        if label < 0 or label >= len(classes):
            raise ValueError(f"{split_name} label {label} out of range for {len(classes)} classes: {record}")

        expected_klass = idx_to_class[label]
        if klass is None:
            klass = expected_klass
        elif klass in class_to_idx and class_to_idx[klass] != label:
            raise ValueError(
                f"{split_name} label/class mismatch for {path_value}: "
                f"label={label}, klass={klass}, expected={expected_klass}"
            )

        out_path = path_value
        if copy_images or link_images:
            rel_source = relative_source_path(path_value, source_path, images_dir, klass)
            rel_target = Path(split_name) / rel_source
            materialize_image(source_path, output_images_dir / rel_target, copy_images, link_images)
            out_path = rel_target.as_posix()

        verification = clean_value(record.get("verification_label"))
        rows.append({
            "file_name": out_path,
            "klass": klass,
            "label": str(label),
            "verification_label": verification if verification is not None else "-1",
        })

    return rows, missing


def write_tsv(path: Path, rows: Sequence[Dict[str, str]], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file without --overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def write_classes(path: Path, classes: Sequence[str], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file without --overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(classes) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="data/Food-101N", help="Destination canonical Food-101N root.")
    parser.add_argument("--train-list", default="", help="Source train split metadata. If omitted, scans train images.")
    parser.add_argument("--test-list", default="", help="Source test split metadata. If omitted, scans test images.")
    parser.add_argument("--task-dir", default="", help="NTD/PuriDivER-style tasks/Food-101N directory.")
    parser.add_argument("--task-dataset", default="Food-101N", help="Dataset prefix used by task JSON files.")
    parser.add_argument("--task-exp", default="blurry10", help="Train task experiment name used by task JSON files.")
    parser.add_argument("--task-rand", type=int, default=1, help="Task split random id, usually 1, 2, or 3.")
    parser.add_argument("--task-classes-per-task", type=int, default=20)
    parser.add_argument("--task-count", type=int, default=5)
    parser.add_argument("--train-images-dir", required=True, help="Source train image directory.")
    parser.add_argument("--test-images-dir", default="", help="Source test image directory. Defaults to train image dir.")
    parser.add_argument("--classes-file", default="", help="Optional source class names file.")
    parser.add_argument("--expected-train-records", type=int, default=EXPECTED_TRAIN_RECORDS)
    parser.add_argument("--expected-test-records", type=int, default=EXPECTED_TEST_RECORDS)
    parser.add_argument("--allow-count-mismatch", action="store_true")
    parser.add_argument("--strict", action="store_true", help="Fail on missing image paths.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output metadata files.")

    materialize = parser.add_mutually_exclusive_group()
    materialize.add_argument("--copy-images", action="store_true", help="Copy selected images into output-root/images.")
    materialize.add_argument("--link-images", action="store_true", help="Symlink selected images into output-root/images.")

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    output_root = resolve_path(args.output_root)
    assert output_root is not None
    train_list = resolve_path(args.train_list)
    test_list = resolve_path(args.test_list)
    task_dir = resolve_path(args.task_dir)
    train_images_dir = resolve_path(args.train_images_dir)
    test_images_dir = resolve_path(args.test_images_dir) or train_images_dir
    classes_file = resolve_path(args.classes_file)
    assert train_images_dir is not None and test_images_dir is not None

    if task_dir is not None:
        train_records = load_task_records(
            task_dir,
            "train",
            args.task_dataset,
            args.task_exp,
            args.task_rand,
            args.task_classes_per_task,
            args.task_count,
        )
        test_records = load_task_records(
            task_dir,
            "test",
            args.task_dataset,
            args.task_exp,
            args.task_rand,
            args.task_classes_per_task,
            args.task_count,
        )
    else:
        train_records = load_records("train", train_list, train_images_dir)
        test_records = load_records("test", test_list, test_images_dir)

    classes = load_classes_file(classes_file)
    if classes is None:
        classes = infer_classes([*train_records, *test_records])

    if len(classes) != 101:
        raise ValueError(f"Expected 101 classes, found {len(classes)}")

    errors = []
    if args.expected_train_records > 0 and len(train_records) != args.expected_train_records:
        errors.append(f"train records={len(train_records)}, expected={args.expected_train_records}")
    if args.expected_test_records > 0 and len(test_records) != args.expected_test_records:
        errors.append(f"test records={len(test_records)}, expected={args.expected_test_records}")
    if errors and not args.allow_count_mismatch:
        print("[food101n-prepare] count check failed:")
        for error in errors:
            print(f"  - {error}")
        print("[food101n-prepare] pass --allow-count-mismatch only for non-formal smoke checks.")
        return 1

    output_meta_dir = output_root / "meta"
    output_images_dir = output_root / "images"

    train_rows, train_missing = normalize_records(
        "train",
        train_records,
        train_images_dir,
        classes,
        args.strict,
        output_images_dir,
        args.copy_images,
        args.link_images,
    )
    test_rows, test_missing = normalize_records(
        "test",
        test_records,
        test_images_dir,
        classes,
        args.strict,
        output_images_dir,
        args.copy_images,
        args.link_images,
    )

    write_tsv(output_meta_dir / "train.tsv", train_rows, args.overwrite)
    write_tsv(output_meta_dir / "test.tsv", test_rows, args.overwrite)
    write_classes(output_meta_dir / "classes.txt", classes, args.overwrite)

    print(f"[food101n-prepare] wrote: {output_meta_dir / 'train.tsv'} ({len(train_rows)} rows)")
    print(f"[food101n-prepare] wrote: {output_meta_dir / 'test.tsv'} ({len(test_rows)} rows)")
    print(f"[food101n-prepare] wrote: {output_meta_dir / 'classes.txt'} ({len(classes)} classes)")
    print(f"[food101n-prepare] missing images: train={train_missing}, test={test_missing}")

    if args.copy_images or args.link_images:
        print("[food101n-prepare] validation command:")
        print(
            "python scripts/validate_food101n_dataset.py "
            f"--root {output_root} --train-list meta/train.tsv --test-list meta/test.tsv "
            "--images-dir images --classes-file meta/classes.txt --strict"
        )
    else:
        print("[food101n-prepare] validation command:")
        print(
            "python scripts/validate_food101n_dataset.py "
            f"--root {output_root} --train-list meta/train.tsv --test-list meta/test.tsv "
            f"--train-images-dir {train_images_dir} --test-images-dir {test_images_dir} "
            "--classes-file meta/classes.txt --strict"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
