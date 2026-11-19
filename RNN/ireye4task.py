import math
import re
import time
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Normalize, v2


task_actions = {
    'cogL': {'Task'}, 'cogH': {'Task'},
    'perL': {'Task'}, 'perH': {'Task'},
    'phyL': {'Task'}, 'phyH': {'Task'},
    'comL': {'lisnH', 'spkL'}, 'comH': {'spkH', 'lisnL'},
}
load_map = {
    'cogL': 0, 'perL': 0, 'phyL': 0, 'comL': 0,
    'cogH': 1, 'perH': 1, 'phyH': 1, 'comH': 1,
}
num_landmarks = 28
pupil_indices = list(range(20, 28))
frame_rate = 60.0
frame_name_pattern = re.compile(r"frame_(\d+)\.jpg")


def resize_with_padding(image, target_size):
    """Resize an image to the target shape without changing its aspect ratio."""
    target_h, target_w = target_size
    h, w = image.shape[:2]
    scale = min(target_w / w, target_h / h)
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    image = cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=interpolation)

    pad_h, pad_w = target_h - image.shape[0], target_w - image.shape[1]
    top, left = pad_h // 2, pad_w // 2
    return cv2.copyMakeBorder(
        image, top, pad_h - top, left, pad_w - left,
        cv2.BORDER_CONSTANT, value=int(image.mean()),
    )


def pupil_eq_diameter(landmarks):
    """Geometric mean of the axes of the ellipse fitted on the pupil landmarks (NaN if invalid)."""
    lm = np.asarray(landmarks, dtype=np.float32)
    points = np.stack([lm[:num_landmarks], lm[num_landmarks:2 * num_landmarks]], axis=1)[pupil_indices]
    if not np.isfinite(points).all():
        return 0.0
    _, axes, _ = cv2.fitEllipse(points)
    return float(np.sqrt(axes[0] * axes[1]))


def find_task_segments(blocks, is_task):
    """Return (start, end) row ranges, end excluded, of each complete task.

    A task starts where the previous row was not a task action and the current one is,
    and ends where the previous row was a task action and the current one is not.
    A change of task block also closes the current task, so a task never mixes labels.
    """
    segments = []
    start = None
    for i, curr_is_task in enumerate(is_task):
        block_changed = i > 0 and blocks[i] != blocks[i - 1]
        if start is not None and (not curr_is_task or block_changed):
            segments.append((start, i))
            start = None
        if curr_is_task and start is None:
            start = i
    if start is not None:
        segments.append((start, len(is_task)))
    return segments


def group_segments_by_block(segments, blocks):
    """Merge consecutive trials of the same block (e.g. the 5 cogH trials before its rating)."""
    groups = []
    for start, end in segments:
        if groups and blocks[groups[-1][-1][0]] == blocks[start]:
            groups[-1].append((start, end))
        else:
            groups.append([(start, end)])
    return groups


def split_into_instances(x, instance_length, pad_value=0.0):
    """Split x along dim 0 into instances of exactly `instance_length` steps.

    x: (T, ...) -> (K, instance_length, ...) with K = ceil(T / instance_length).
    If T is not a multiple of instance_length, the last instance is completed with
    pad_value. If T == instance_length, the result is x itself with K = 1.
    Also returns a boolean mask (K, instance_length) that is True on real steps.
    """
    num_steps = x.shape[0]
    num_instances = max(1, math.ceil(num_steps / instance_length))
    pad = num_instances * instance_length - num_steps
    if pad:
        x = F.pad(x, [0, 0] * (x.dim() - 1) + [0, pad], value=pad_value)
    mask = torch.arange(num_instances * instance_length) < num_steps
    return x.unflatten(0, (num_instances, instance_length)), mask.view(num_instances, instance_length)


class IREye4Task(Dataset):
    """Task samples from IREye4Task as bags for multiple instance learning.

    group_by="trial": one bag per trial (e.g. each preTask -> Task -> pause cycle).
    group_by="block": one bag per block, i.e. all consecutive trials of the same
    block concatenated (pauses and preTask excluded), so 16 bags per subject.

    Each bag is split into K instances of exactly `instance_length` frames (counted
    after frame_stride); the last instance is padded when the bag length is not a
    multiple of instance_length. The label is one per bag.
    """

    def __init__(
        self,
        dataset_dir,
        subject_ids=None,
        group_by="block",
        frame_stride=1,
        min_frames=1,
        instance_length=32,
        target_size=(224, 224),
        load_frames=True,
        only_cognitive=False,
        rgb_format=False,
        transform=None,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.subject_ids = {str(subject) for subject in subject_ids} if subject_ids is not None else None
        if group_by not in ("trial", "block"):
            raise ValueError("group_by must be 'trial' or 'block'")
        self.group_by = group_by
        self.frame_stride = frame_stride
        self.min_frames = min_frames
        self.instance_length = instance_length
        self.target_size = target_size
        self.load_frames = load_frames
        self.only_cognitive = only_cognitive
        self.rgb_format = rgb_format
        self.transform = transform
        self.normalize = (
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            if self.rgb_format else Normalize(mean=[0.485], std=[0.229])
        )
        self.samples = []

        if self.frame_stride <= 0 or self.min_frames <= 0 or self.instance_length <= 0:
            raise ValueError("frame_stride, min_frames and instance_length must be positive")
        if not self.dataset_dir.is_dir():
            raise FileNotFoundError(f"Dataset directory does not exist: {self.dataset_dir}")

        self._index_dataset()

    def _find_frame_directory(self, subject_id):
        for eye in ("eye0", "eye1"):
            frame_dir = self.dataset_dir / f"{eye}_sub{subject_id}"
            if frame_dir.is_dir():
                return frame_dir
        return None

    def _get_frame_count(self, frame_dir):
        frame_count = 0
        max_index = -1
        for frame_path in frame_dir.iterdir():
            match = frame_name_pattern.fullmatch(frame_path.name)
            if match:
                frame_count += 1
                max_index = max(max_index, int(match.group(1)))

        if frame_count == 0:
            raise ValueError(f"No frame_*.jpg images found in {frame_dir}")
        if frame_count != max_index + 1:
            raise ValueError(f"Frame files in {frame_dir} are not a continuous sequence from frame 0")
        return frame_count

    def _index_dataset(self):
        found_subjects = set()
        for csv_path in sorted(self.dataset_dir.glob("tasks_sub*.csv")):
            subject_id = csv_path.stem.removeprefix("tasks_sub")
            if self.subject_ids is not None and subject_id not in self.subject_ids:
                continue

            frame_dir = self._find_frame_directory(subject_id)
            if frame_dir is None:
                continue

            eye_name = frame_dir.name.split("_")[0]
            annotation_path = self.dataset_dir / f"{eye_name}_sub{subject_id}.txt"
            if not annotation_path.is_file():
                raise FileNotFoundError(f"Missing frame timestamps: {annotation_path}")

            task_df = pd.read_csv(csv_path)
            if task_df.shape[1] < 2:
                raise ValueError(f"Expected task block and action columns in {csv_path}")

            annotations = pd.read_csv(annotation_path, sep=r"\s+", skiprows=1, header=None)
            timestamps = annotations.iloc[:, 0].to_numpy(dtype=np.float64)
            landmarks = annotations.drop(columns=[0, 1]).to_numpy(dtype=np.float32)
            if landmarks.shape[1] != 2 * num_landmarks:
                raise ValueError(
                    f"Expected {2 * num_landmarks} landmark columns in {annotation_path}, found {landmarks.shape[1]}"
                )
            if len(timestamps) != len(task_df):
                raise ValueError(
                    f"Task rows ({len(task_df)}) do not match timestamp rows ({len(timestamps)}) for subject {subject_id}"
                )
            if not np.isfinite(timestamps).all():
                raise ValueError(f"Non-finite frame timestamp in {annotation_path}")

            frame_indices = np.rint(timestamps * frame_rate).astype(np.int64)
            if (frame_indices < 0).any() or (np.abs(timestamps * frame_rate - frame_indices) > 0.1).any():
                raise ValueError(f"Frame timestamps do not map to frame indices in {annotation_path}")

            frame_count = self._get_frame_count(frame_dir)
            if (frame_indices >= frame_count).any():
                raise ValueError(f"Timestamp references a missing frame in {frame_dir} (frame count: {frame_count})")

            found_subjects.add(subject_id)
            blocks = task_df.iloc[:, 0].astype(str).str.strip().to_numpy()
            actions = task_df.iloc[:, 1].astype(str).str.strip().to_numpy()
            is_task = np.array([action in task_actions.get(block, ()) for block, action in zip(blocks, actions)])
            if self.only_cognitive:
                is_task &= np.array([block.startswith("cog") for block in blocks])

            trials = find_task_segments(blocks, is_task)
            for start, end in trials:
                gaps = int((np.diff(frame_indices[start:end]) != 1).sum())
                if gaps:
                    warnings.warn(
                        f"Subject {subject_id}, task {blocks[start]} (rows {start}-{end - 1}): "
                        f"{gaps} non-contiguous frame step(s)"
                    )

            if self.group_by == "trial":
                groups = [[trial] for trial in trials]
            else:
                groups = group_segments_by_block(trials, blocks)

            for group in groups:
                block = blocks[group[0][0]]
                task_rows = np.concatenate([np.arange(start, end) for start, end in group])
                rows = task_rows[::self.frame_stride]
                if len(rows) < self.min_frames:
                    continue

                diameters = np.array([pupil_eq_diameter(landmarks[r]) for r in rows], dtype=np.float32)
                self.samples.append({
                    "subject": subject_id,
                    "block": block,
                    "label": load_map[block],
                    "num_trials": len(group),
                    "num_instances": math.ceil(len(rows) / self.instance_length),
                    "frame_dir": frame_dir,
                    "frame_indices": frame_indices[rows],
                    "diameters": diameters,
                })

        if self.subject_ids is not None:
            missing = self.subject_ids - found_subjects
            if missing:
                raise ValueError(f"Subjects not found in {self.dataset_dir}: {sorted(missing)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """Return one bag.

        video:      (K, C, m, H, W), or an empty tensor if load_frames=False
        diameters:  (K, m, 1)
        frame_mask: (K, m), True on real frames, False on padding
        label:      int, one per bag
        """
        sample = self.samples[idx]
        m = self.instance_length
        diameters = torch.from_numpy(sample["diameters"]).unsqueeze(-1)            # (T, 1)
        diameters, frame_mask = split_into_instances(diameters, m)                  # (K, m, 1), (K, m)
        label = torch.tensor(sample["label"])
        video_tensor = torch.empty(0)

        if self.load_frames:
            frames = []
            for frame_idx in sample["frame_indices"]:
                frame_path = sample["frame_dir"] / f"frame_{frame_idx:06d}.jpg"
                gray = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
                if gray is None:
                    raise OSError(f"Could not read frame: {frame_path}")
                if self.target_size:
                    gray = resize_with_padding(gray, self.target_size)
                frames.append(gray)

            video_tensor = torch.from_numpy(np.stack(frames)).to(torch.float32).unsqueeze(1) / 255.0
            if self.rgb_format:
                video_tensor = video_tensor.repeat(1, 3, 1, 1)
            if self.transform:
                video_tensor = self.transform(video_tensor)
            video_tensor = self.normalize(video_tensor)                                # (T, C, H, W)
            # Padding frames are 0 after normalization, i.e. the normalization mean.
            video_tensor, _ = split_into_instances(video_tensor, m)                    # (K, m, C, H, W)
            video_tensor = video_tensor.permute(0, 2, 1, 3, 4)                         # (K, C, m, H, W)

        return video_tensor, diameters, frame_mask, label


def mil_collate(batch):
    """Pad bags with a different number of instances to the largest one in the batch.

    Returns:
      videos:        (B, K, C, m, H, W), or an empty tensor if frames are not loaded
      diameters:     (B, K, m, 1)
      labels:        (B,) one label per bag
      frame_mask:    (B, K, m), True on real frames
      instance_mask: (B, K), True on real instances (False on instances added to fill the batch)
    """
    videos, diameters, frame_masks, labels = zip(*batch)
    num_instances = torch.tensor([d.shape[0] for d in diameters])
    diameters = pad_sequence(diameters, batch_first=True)
    frame_mask = pad_sequence(frame_masks, batch_first=True, padding_value=False)
    instance_mask = torch.arange(diameters.shape[1])[None, :] < num_instances[:, None]
    labels = torch.tensor(labels, dtype=torch.long)

    if videos[0].numel() == 0:
        videos = torch.empty(0)
    else:
        videos = pad_sequence(videos, batch_first=True)
    return videos, diameters, labels, frame_mask, instance_mask


def get_subject_splits():
    """Return subject-level train, validation, and test splits."""
    train_subjects = list(range(1, 13))
    val_subjects = [13, 14, 16]
    test_subjects = [17, 18, 19]
    return train_subjects, val_subjects, test_subjects


DATASET_PATH = "/work/cvcs2026/LZMM/IREye4Task"


def get_dataloaders(
    dataset_path=DATASET_PATH,
    group_by="block",
    frame_stride=5,
    min_frames=1,
    instance_length=32,
    load_frames=False,
    only_cognitive=False,
    batch_size=1,
    num_workers=4,
    verbose=True,
):
    """Build train/val/test datasets and DataLoaders. Returns (train_dl, val_dl, test_dl)."""
    start_time = time.perf_counter()
    train_subs, val_subs, test_subs = get_subject_splits()

    train_transforms = v2.Compose([
        v2.RandomResizedCrop(size=(224, 224), scale=(0.85, 1.0), ratio=(1.0, 1.0), antialias=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ColorJitter(brightness=0.2, contrast=0.2),
    ])
    val_test_transforms = v2.Compose([
        v2.Resize((224, 224), antialias=True),
    ])

    common = dict(
        dataset_dir=dataset_path,
        group_by=group_by,
        frame_stride=frame_stride,
        min_frames=min_frames,
        instance_length=instance_length,
        load_frames=load_frames,
        only_cognitive=only_cognitive,
        rgb_format=True,
    )
    train_dataset = IREye4Task(subject_ids=train_subs, transform=train_transforms, **common)
    val_dataset = IREye4Task(subject_ids=val_subs, transform=val_test_transforms, **common)
    test_dataset = IREye4Task(subject_ids=test_subs, transform=val_test_transforms, **common)

    loader_args = dict(batch_size=batch_size, num_workers=num_workers, collate_fn=mil_collate)
    train_dl = DataLoader(train_dataset, shuffle=True, **loader_args)
    val_dl = DataLoader(val_dataset, **loader_args)
    test_dl = DataLoader(test_dataset, **loader_args)

    if verbose:
        print(f"Dataset loaded in {time.perf_counter() - start_time:.3f} s")
        for name, ds in (("Train", train_dataset), ("Validation", val_dataset), ("Test", test_dataset)):
            lengths = [len(s["frame_indices"]) for s in ds.samples]
            bags = [s["num_instances"] for s in ds.samples]
            print(f"{name}: {len(ds)} bags, frames per bag min/median/max = "
                  f"{min(lengths)}/{int(np.median(lengths))}/{max(lengths)}, "
                  f"instances of {instance_length} frames per bag min/median/max = "
                  f"{min(bags)}/{int(np.median(bags))}/{max(bags)}")

    return train_dl, val_dl, test_dl


if __name__ == "__main__":
    get_dataloaders(group_by="trial")