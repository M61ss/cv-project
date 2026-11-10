import re
import time
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
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
        return np.nan
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


class IREye4Task(Dataset):
    """One sample per complete task of IREye4Task, with the load label repeated on every frame."""

    def __init__(
        self,
        dataset_dir,
        subject_ids=None,
        frame_stride=1,
        min_frames=1,
        target_size=(224, 224),
        load_frames=True,
        only_cognitive=False,
        rgb_format=False,
        transform=None,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.subject_ids = {str(subject) for subject in subject_ids} if subject_ids is not None else None
        self.frame_stride = frame_stride
        self.min_frames = min_frames
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

        if self.frame_stride <= 0 or self.min_frames <= 0:
            raise ValueError("frame_stride and min_frames must be positive")
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

            for start, end in find_task_segments(blocks, is_task):
                block = blocks[start]
                segment_frames = frame_indices[start:end]
                gaps = int((np.diff(segment_frames) != 1).sum())
                if gaps:
                    warnings.warn(
                        f"Subject {subject_id}, task {block} (rows {start}-{end - 1}): "
                        f"{gaps} non-contiguous frame step(s)"
                    )

                rows = np.arange(start, end, self.frame_stride)
                if len(rows) < self.min_frames:
                    continue

                diameters = np.array([pupil_eq_diameter(landmarks[r]) for r in rows], dtype=np.float32)
                self.samples.append({
                    "subject": subject_id,
                    "block": block,
                    "label": load_map[block],
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
        sample = self.samples[idx]
        num_frames = len(sample["frame_indices"])
        diameters_tensor = torch.from_numpy(sample["diameters"]).unsqueeze(-1)            # (T, 1)
        labels_tensor = torch.full((num_frames, 1), sample["label"], dtype=torch.float32)  # (T, 1)
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
            video_tensor = self.normalize(video_tensor).permute(1, 0, 2, 3)               # (C, T, H, W)

        return video_tensor, diameters_tensor, labels_tensor


def pad_collate(batch):
    """Pad tasks of different length to the longest one in the batch.

    Returns video (B, C, T, H, W) or an empty tensor, diameters (B, T, 1),
    labels (B, T, 1) padded with -1, and a boolean mask (B, T) that is True on real frames.
    """
    videos, diameters, labels = zip(*batch)
    lengths = torch.tensor([d.shape[0] for d in diameters])
    diameters = pad_sequence(diameters, batch_first=True)
    labels = pad_sequence(labels, batch_first=True, padding_value=-1.0)
    mask = torch.arange(diameters.shape[1])[None, :] < lengths[:, None]

    if videos[0].numel() == 0:
        videos = torch.empty(0)
    else:
        videos = pad_sequence([v.permute(1, 0, 2, 3) for v in videos], batch_first=True)  # (B, T, C, H, W)
        videos = videos.permute(0, 2, 1, 3, 4)
    return videos, diameters, labels, mask


def get_subject_splits():
    """Return subject-level train, validation, and test splits."""
    train_subjects = list(range(1, 13))
    val_subjects = [13, 14, 16]
    test_subjects = [17, 18, 19]
    return train_subjects, val_subjects, test_subjects


if __name__ == "__main__":
    print('Loading dataset...')
    print('')
    start_time = time.perf_counter()

    ds_config = {
        'frame_stride': 4,
        'min_frames': 16,
    }
    for k, v in ds_config.items():
        print(f"{k}: {v}")

    dataset_path = "/work/cvcs2026/LZMM/IREye4Task"
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
        rgb_format=True, 
        load_frames=False, 
        **ds_config
    )
    train_dataset = IREye4Task(subject_ids=train_subs, transform=train_transforms, **common)
    val_dataset = IREye4Task(subject_ids=val_subs, transform=val_test_transforms, **common)
    test_dataset = IREye4Task(subject_ids=test_subs, transform=val_test_transforms, **common)

    loader_args = dict(
        batch_size=1, 
        num_workers=4, 
        collate_fn=pad_collate
    )
    train_dl = DataLoader(train_dataset, shuffle=True, **loader_args)
    val_dl = DataLoader(val_dataset, **loader_args)
    test_dl = DataLoader(test_dataset, **loader_args)

    end_time = time.perf_counter()
    print('Dataset loaded!')
    print(f'Elapsed time: {end_time - start_time:.3f} s')
    print('')

    for name, ds in (("Train", train_dataset), ("Validation", val_dataset), ("Test", test_dataset)):
        lengths = [len(s["frame_indices"]) for s in ds.samples]
        print(f'{name}: {len(ds)} tasks, frames per task min/median/max = '
              f'{min(lengths)}/{int(np.median(lengths))}/{max(lengths)}')