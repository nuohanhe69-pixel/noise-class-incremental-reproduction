import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import InterpolationMode

from datasets.transforms.denormalization import DeNormalize
from datasets.utils import set_default_from_args
from datasets.utils.continual_dataset import (
    ContinualDataset,
    fix_class_names_order,
    store_masked_loaders,
)
from utils.conf import base_path
from utils.prompt_templates import templates


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PATH_KEYS = ("file_name", "filepath", "file_path", "path", "image", "img", "sample_key")
CLASS_KEYS = ("klass", "class", "class_name", "category", "category_name", "label_name")
LABEL_KEYS = ("label", "target", "class_idx", "class_id", "category_id")
VERIFICATION_KEYS = ("verification_label", "verified", "is_verified")


def _clean_value(value: Any) -> Optional[str]:
    if value is None:
        return None
    value = str(value).strip()
    if value == "" or value.lower() in {"none", "nan", "null"}:
        return None
    return value


def _is_int_like(value: Any) -> bool:
    value = _clean_value(value)
    if value is None:
        return False
    try:
        int(value)
    except ValueError:
        return False
    return True


def _first_value(row: Dict[str, Any], keys: Sequence[str]) -> Optional[str]:
    lowered = {str(k).strip().lower(): v for k, v in row.items()}
    for key in keys:
        value = _clean_value(lowered.get(key))
        if value is not None:
            return value
    return None


def _normalise_dict_record(row: Dict[str, Any]) -> Dict[str, Any]:
    record = dict(row)
    path = _first_value(record, PATH_KEYS)
    klass = _first_value(record, CLASS_KEYS)
    label = _first_value(record, LABEL_KEYS)
    verification = _first_value(record, VERIFICATION_KEYS)

    out: Dict[str, Any] = {}
    if path is not None:
        out["file_name"] = path
    if klass is not None:
        out["klass"] = klass
    if label is not None:
        out["label"] = label
    if verification is not None:
        out["verification_label"] = verification
    return out


def _normalise_sequence_record(parts: Sequence[str]) -> Dict[str, Any]:
    record: Dict[str, Any] = {"file_name": parts[0]}
    if len(parts) > 1:
        if _is_int_like(parts[1]):
            record["label"] = parts[1]
        else:
            record["klass"] = parts[1]
    if len(parts) > 2:
        if "klass" not in record and not _is_int_like(parts[2]):
            record["klass"] = parts[2]
        elif _is_int_like(parts[2]):
            record["verification_label"] = parts[2]
    return record


def _read_json_records(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, list):
        records = []
        for item in raw:
            if isinstance(item, dict):
                records.append(_normalise_dict_record(item))
            else:
                records.append({"file_name": str(item)})
        return records

    if not isinstance(raw, dict):
        raise ValueError(f"Unsupported JSON metadata format in {path}")

    for key in ("annotations", "samples", "data", "images", "items"):
        value = raw.get(key)
        if isinstance(value, list):
            return [
                _normalise_dict_record(item) if isinstance(item, dict) else {"file_name": str(item)}
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


def _read_text_records(path: Path) -> List[Dict[str, Any]]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    lines = [line for line in lines if line and not line.startswith("#")]
    if not lines:
        return []

    first = lines[0].lower()
    delimiter = "\t" if "\t" in lines[0] else "," if "," in lines[0] else None
    header_tokens = set(first.replace("\t", ",").replace(" ", ",").split(","))
    known_keys = set(PATH_KEYS + CLASS_KEYS + LABEL_KEYS + VERIFICATION_KEYS)

    if delimiter is not None and header_tokens.intersection(known_keys):
        reader = csv.DictReader(lines, delimiter=delimiter)
        return [_normalise_dict_record(row) for row in reader]

    records = []
    for line in lines:
        parts = line.split(delimiter) if delimiter is not None else line.split()
        parts = [_clean_value(part) for part in parts]
        parts = [part for part in parts if part is not None]
        if parts:
            records.append(_normalise_sequence_record(parts))
    return records


def _read_metadata_records(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".json":
        return _read_json_records(path)
    return _read_text_records(path)


def _load_classes_file(path: Optional[Path]) -> Optional[List[str]]:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"Food101N classes file not found: {path}")

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
        if len(parts) > 1 and _is_int_like(parts[0]):
            classes.append(parts[1])
        else:
            classes.append(parts[0])
    return classes


def _record_class_name(record: Dict[str, Any]) -> Optional[str]:
    klass = _clean_value(record.get("klass"))
    if klass is not None:
        return klass
    path = _clean_value(record.get("file_name"))
    if path is None:
        return None
    parts = Path(path).parts
    return parts[0] if len(parts) > 1 else None


def _record_label(record: Dict[str, Any]) -> Optional[int]:
    label = _clean_value(record.get("label"))
    return int(label) if _is_int_like(label) else None


def _record_verification(record: Dict[str, Any]) -> Optional[int]:
    label = _clean_value(record.get("verification_label"))
    return int(label) if _is_int_like(label) else None


def _classes_from_records(records: Sequence[Dict[str, Any]]) -> List[str]:
    names = set()
    label_to_name: Dict[int, str] = {}
    label_values = []

    for record in records:
        klass = _record_class_name(record)
        label = _record_label(record)
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

    raise ValueError("Could not infer Food101N class names from metadata.")


def _scan_image_records(images_dir: Path) -> List[Dict[str, Any]]:
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


def _suffix_candidates(path: Path) -> Iterable[Path]:
    if path.suffix:
        yield path
        return
    for suffix in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
        yield path.with_suffix(suffix)


def _resolve_image_path(path_value: str, images_dir: Path) -> Path:
    path = Path(path_value)
    base = path if path.is_absolute() else images_dir / path
    for candidate in _suffix_candidates(base):
        if candidate.exists():
            return candidate
    return next(iter(_suffix_candidates(base)))


def _first_existing(paths: Sequence[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists():
            return path
    return None


def _default_metadata_path(root: Path, train: bool) -> Optional[Path]:
    names = (
        ("meta/train.json", "meta/train.txt", "meta/train.tsv", "meta/train.csv",
         "train.json", "train.txt", "train.tsv", "train.csv")
        if train else
        ("meta/test.json", "meta/test.txt", "meta/test.tsv", "meta/test.csv",
         "test.json", "test.txt", "test.tsv", "test.csv")
    )
    return _first_existing([root / name for name in names])


def _default_images_dir(root: Path, train: bool) -> Path:
    names = (
        ("train", "Food-101N_release/train", "images", ".")
        if train else
        ("test", "val", "images", "Food-101/images", "food-101/images", ".")
    )
    existing = _first_existing([root / name for name in names])
    return existing if existing is not None else root / ("train" if train else "test")


def _resolve_root(user_root: Optional[str]) -> Path:
    if user_root:
        path = Path(user_root).expanduser()
        return path if path.is_absolute() else Path(base_path()) / path

    candidates = [Path(base_path()) / name for name in (
        "Food-101N",
        "Food-101N_release",
        "food-101n",
        "Food101N",
        "food101n",
    )]
    existing = _first_existing(candidates)
    return existing if existing is not None else candidates[0]


def _resolve_optional_path(value: Optional[str], root: Path) -> Optional[Path]:
    if value is None or value == "":
        return None
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    root_candidate = root / path
    base_candidate = Path(base_path()) / path
    if root_candidate.exists() or not base_candidate.exists():
        return root_candidate
    return base_candidate


class Food101N(Dataset):
    """Food-101N image dataset with noisy train labels and clean Food-101 test labels."""

    def __init__(
        self,
        root: Path,
        train: bool = True,
        transform=None,
        target_transform=None,
        metadata_path: Optional[Path] = None,
        images_dir: Optional[Path] = None,
        classes_file: Optional[Path] = None,
        class_to_idx: Optional[Dict[str, int]] = None,
        strict: bool = False,
    ) -> None:
        self.root = Path(root)
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.metadata_path = metadata_path if metadata_path is not None else _default_metadata_path(self.root, train)
        self.images_dir = images_dir if images_dir is not None else _default_images_dir(self.root, train)
        self.not_aug_transform = transforms.Compose([
            transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])

        if self.metadata_path is not None:
            if not self.metadata_path.exists():
                raise FileNotFoundError(f"Food101N metadata file not found: {self.metadata_path}")
            records = _read_metadata_records(self.metadata_path)
        else:
            records = _scan_image_records(self.images_dir)

        if not records:
            expected = (
                f"Expected a metadata file under {self.root}/meta or images arranged as "
                f"{self.images_dir}/<class_name>/<image>."
            )
            raise FileNotFoundError(f"No Food101N samples found. {expected}")

        classes = _load_classes_file(classes_file)
        if class_to_idx is None:
            classes = classes if classes is not None else _classes_from_records(records)
            self.classes = list(classes)
            self.class_to_idx = {klass: idx for idx, klass in enumerate(self.classes)}
        else:
            self.class_to_idx = dict(class_to_idx)
            self.classes = [None] * len(self.class_to_idx)
            for klass, idx in self.class_to_idx.items():
                self.classes[idx] = klass

        self.idx_to_class = {idx: klass for klass, idx in self.class_to_idx.items()}

        data: List[str] = []
        targets: List[int] = []
        verification_labels: List[int] = []
        missing_paths: List[Path] = []

        for record in records:
            path_value = _clean_value(record.get("file_name"))
            if path_value is None:
                raise ValueError(f"Food101N record has no image path: {record}")

            image_path = _resolve_image_path(path_value, self.images_dir)
            if not image_path.exists():
                missing_paths.append(image_path)

            label = _record_label(record)
            klass = _record_class_name(record)
            if label is None:
                if klass is None or klass not in self.class_to_idx:
                    raise ValueError(f"Cannot resolve Food101N label for record: {record}")
                label = self.class_to_idx[klass]
            elif label < 0 or label >= len(self.classes):
                raise ValueError(f"Food101N label {label} out of range for {len(self.classes)} classes.")

            if klass is not None and klass in self.class_to_idx and self.class_to_idx[klass] != label:
                raise ValueError(
                    f"Food101N label/class mismatch for {path_value}: label={label}, "
                    f"class={klass}, class_to_idx={self.class_to_idx[klass]}"
                )

            data.append(str(image_path))
            targets.append(label)
            verification = _record_verification(record)
            verification_labels.append(-1 if verification is None else verification)

        if strict and missing_paths:
            examples = "\n".join(str(path) for path in missing_paths[:10])
            raise FileNotFoundError(
                f"Food101N strict path check found {len(missing_paths)} missing images. "
                f"First missing paths:\n{examples}"
            )

        if missing_paths:
            logging.warning("Food101N found %d missing image paths; first one: %s", len(missing_paths), missing_paths[0])

        self.data = np.array(data, dtype=object)
        self.targets = np.array(targets, dtype=np.int64)
        self.verification_labels = np.array(verification_labels, dtype=np.int64)
        self.noisy_targets = self.targets.copy() if self.train else None
        self.clean_targets = None if self.train else self.targets.copy()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        image_path = Path(self.data[index])
        if not image_path.exists():
            raise FileNotFoundError(f"Food101N image not found: {image_path}")

        target = int(self.targets[index])
        image = Image.open(image_path).convert("RGB")
        not_aug_img = self.not_aug_transform(image.copy())

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target, not_aug_img


class SequentialFood101N(ContinualDataset):
    NAME = "seq-food101n"
    SETTING = "class-il"
    N_CLASSES = 101
    N_TASKS = 5
    N_CLASSES_PER_TASK = [20, 20, 20, 20, 21]
    SIZE = (224, 224)
    MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

    TRANSFORM = transforms.Compose([
        transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    TEST_TRANSFORM = transforms.Compose([
        transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    def __init__(
        self,
        args,
        food101n_root: str = "",
        food101n_train_list: str = "",
        food101n_test_list: str = "",
        food101n_images_dir: str = "",
        food101n_train_images_dir: str = "",
        food101n_test_images_dir: str = "",
        food101n_classes_file: str = "",
        food101n_strict: int = 0,
    ) -> None:
        super().__init__(args)
        if getattr(args, "permute_classes", False) and hasattr(args, "class_order"):
            args.class_order = np.asarray(args.class_order)

        self.food101n_root = _resolve_root(food101n_root)
        self.food101n_train_list = _resolve_optional_path(food101n_train_list, self.food101n_root)
        self.food101n_test_list = _resolve_optional_path(food101n_test_list, self.food101n_root)
        self.food101n_images_dir = _resolve_optional_path(food101n_images_dir, self.food101n_root)
        self.food101n_train_images_dir = _resolve_optional_path(food101n_train_images_dir, self.food101n_root)
        self.food101n_test_images_dir = _resolve_optional_path(food101n_test_images_dir, self.food101n_root)
        self.food101n_classes_file = _resolve_optional_path(food101n_classes_file, self.food101n_root)
        self.food101n_strict = bool(food101n_strict)
        self._loaded_class_names: Optional[List[str]] = None

    def get_data_loaders(self):
        if getattr(self.args, "noise_rate", 0) not in (0, 0.0):
            raise ValueError(
                "Food101N already uses real noisy training labels. Do not set --noise_rate; "
                "the generic synthetic-noise pipeline is intended for datasets such as CIFAR."
            )

        train_images_dir = self.food101n_train_images_dir or self.food101n_images_dir
        test_images_dir = self.food101n_test_images_dir or self.food101n_images_dir

        train_dataset = Food101N(
            self.food101n_root,
            train=True,
            transform=self.TRANSFORM,
            metadata_path=self.food101n_train_list,
            images_dir=train_images_dir,
            classes_file=self.food101n_classes_file,
            strict=self.food101n_strict,
        )

        if len(train_dataset.classes) != self.N_CLASSES:
            raise ValueError(
                f"Food101N expected {self.N_CLASSES} classes, found {len(train_dataset.classes)}. "
                "Provide --food101n_classes_file or a complete train/test metadata layout."
            )

        test_dataset = Food101N(
            self.food101n_root,
            train=False,
            transform=self.TEST_TRANSFORM,
            metadata_path=self.food101n_test_list,
            images_dir=test_images_dir,
            classes_file=self.food101n_classes_file,
            class_to_idx=train_dataset.class_to_idx,
            strict=self.food101n_strict,
        )

        self._loaded_class_names = list(train_dataset.classes)
        return store_masked_loaders(train_dataset, test_dataset, self)

    def get_class_names(self):
        if self.class_names is not None:
            return self.class_names
        if self._loaded_class_names is None:
            classes = _load_classes_file(self.food101n_classes_file)
            if classes is None:
                classes = [f"food_class_{i:03d}" for i in range(self.N_CLASSES)]
        else:
            classes = self._loaded_class_names
        self.class_names = fix_class_names_order(classes, self.args)
        return self.class_names

    @staticmethod
    def get_prompt_templates():
        return templates["imagenet"]

    @staticmethod
    def get_transform():
        return transforms.Compose([transforms.ToPILImage(), SequentialFood101N.TRANSFORM])

    @set_default_from_args("backbone")
    def get_backbone():
        return "resnet18"

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_normalization_transform():
        return transforms.Normalize(SequentialFood101N.MEAN, SequentialFood101N.STD)

    @staticmethod
    def get_denormalization_transform():
        return DeNormalize(SequentialFood101N.MEAN, SequentialFood101N.STD)

    @set_default_from_args("n_epochs")
    def get_epochs(self):
        return 50

    @set_default_from_args("batch_size")
    def get_batch_size(self):
        return 32

    @set_default_from_args("minibatch_size")
    def get_minibatch_size(self):
        return 32
