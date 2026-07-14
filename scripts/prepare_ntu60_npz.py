import argparse
import os
import re
import zipfile

import numpy as np
from tqdm import tqdm


CS_TRAIN_SUBJECTS = {1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38}
CV_TRAIN_CAMERAS = {2, 3}


def parse_name(name):
    base = os.path.basename(name)
    match = re.match(r"S(\d{3})C(\d{3})P(\d{3})R(\d{3})A(\d{3})", base)
    if match is None:
        raise ValueError(f"Unexpected NTU skeleton filename: {base}")
    setup, camera, subject, replication, action = map(int, match.groups())
    return setup, camera, subject, replication, action - 1


def read_skeleton(text, max_frames=300, max_bodies=2, num_joints=25):
    lines = iter(text.splitlines())
    num_frames = int(next(lines))
    data = np.zeros((max_frames, max_bodies, num_joints, 3), dtype=np.float16)

    for frame_idx in range(num_frames):
        num_bodies = int(next(lines))
        bodies = []
        for _ in range(num_bodies):
            next(lines)
            joint_count = int(next(lines))
            joints = np.zeros((num_joints, 3), dtype=np.float16)
            for joint_idx in range(joint_count):
                values = next(lines).split()
                if joint_idx < num_joints:
                    joints[joint_idx] = np.asarray(values[:3], dtype=np.float16)
            bodies.append(joints)

        if frame_idx < max_frames and bodies:
            energy = [np.abs(body).sum() for body in bodies]
            keep = np.argsort(energy)[-max_bodies:][::-1]
            for out_idx, body_idx in enumerate(keep):
                data[frame_idx, out_idx] = bodies[body_idx]

    return data.reshape(max_frames, max_bodies * num_joints * 3)


def one_hot(labels, num_classes=60):
    out = np.zeros((len(labels), num_classes), dtype=np.float32)
    out[np.arange(len(labels)), labels] = 1.0
    return out


def iter_members(path):
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as zf:
            for name in sorted(n for n in zf.namelist() if n.endswith(".skeleton")):
                yield name, zf.read(name).decode("utf-8", errors="ignore")
    else:
        skeleton_files = []
        for root, _, files in os.walk(path):
            skeleton_files.extend(os.path.join(root, file) for file in files if file.endswith(".skeleton"))
        for file in sorted(skeleton_files):
            with open(file, "r", encoding="utf-8", errors="ignore") as handle:
                yield file, handle.read()


def build_npz(source, output, split="cs", max_frames=300):
    x_train, y_train, x_test, y_test = [], [], [], []

    for name, text in tqdm(iter_members(source), desc="Converting NTU60 skeletons"):
        _, camera, subject, _, action = parse_name(name)
        sample = read_skeleton(text, max_frames=max_frames)
        is_train = subject in CS_TRAIN_SUBJECTS if split == "cs" else camera in CV_TRAIN_CAMERAS

        if is_train:
            x_train.append(sample)
            y_train.append(action)
        else:
            x_test.append(sample)
            y_test.append(action)

    np.savez_compressed(
        output,
        x_train=np.asarray(x_train, dtype=np.float16),
        y_train=one_hot(np.asarray(y_train, dtype=np.int64)),
        x_test=np.asarray(x_test, dtype=np.float16),
        y_test=one_hot(np.asarray(y_test, dtype=np.int64)),
    )


def main():
    parser = argparse.ArgumentParser(description="Convert NTU RGB+D 60 .skeleton files to the .npz format used by seq-ntu60.")
    parser.add_argument("--source", required=True, help="Path to nturgbd skeleton zip or extracted directory.")
    parser.add_argument("--output", default="data/NTU60_CS.npz", help="Output .npz path.")
    parser.add_argument("--split", choices=["cs", "cv"], default="cs", help="Official cross-subject or cross-view split.")
    parser.add_argument("--max_frames", type=int, default=300)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    build_npz(args.source, args.output, args.split, args.max_frames)


if __name__ == "__main__":
    main()
