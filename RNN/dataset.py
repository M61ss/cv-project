import os
import re

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Normalize, v2


def resize_with_padding(image, target_size):
    """Resize an image to the target shape without changing its aspect ratio."""
    target_h, target_w = target_size
    height, width = image.shape[:2]
    scale = min(target_w / width, target_h / height)
    resized_w = round(width * scale)
    resized_h = round(height * scale)
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    image = cv2.resize(image, (resized_w, resized_h), interpolation=interpolation)

    top = (target_h - resized_h) // 2
    bottom = target_h - resized_h - top
    left = (target_w - resized_w) // 2
    right = target_w - resized_w - left
    return cv2.copyMakeBorder(
        image,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=int(image.mean()),
    )


class IREye4TaskVideoDataset(Dataset):
    """Load fixed-length labeled clips from extracted IREye4Task eye frames."""

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
    frame_rate = 60.0
    frame_name_pattern = re.compile(r"frame_(\d+)\.jpg$")

    def __init__(
        self,
        dataset_dir,
        subject_ids=None,
        num_frames=16,
        frame_stride=1,
        clip_stride=16,
        target_size=(224, 224),
        only_cognitive=False,
        rgb_format=False,
        transform=None,
    ):
        self.dataset_dir = dataset_dir
        self.subject_ids = {str(subject) for subject in subject_ids} if subject_ids is not None else None
        self.num_frames = num_frames
        self.frame_stride = frame_stride
        self.clip_stride = clip_stride
        self.target_size = target_size
        self.only_cognitive = only_cognitive
        self.rgb_format = rgb_format
        self.transform = transform
        self.samples = []

        if self.num_frames <= 0 or self.frame_stride <= 0 or self.clip_stride <= 0:
            raise ValueError("num_frames, frame_stride, and clip_stride must be positive")
        if not os.path.isdir(self.dataset_dir):
            raise FileNotFoundError(f"Dataset directory does not exist: {self.dataset_dir}")

        self._index_dataset()

    def _get_frame_directory(self, subject_id):
        for eye in ("eye0", "eye1"):
            frame_dir = os.path.join(self.dataset_dir, f"{eye}_sub{subject_id}")
            if os.path.isdir(frame_dir):
                return frame_dir
        return None

    def _get_frame_count(self, frame_dir):
        frame_count = 0
        max_index = -1
        for filename in os.listdir(frame_dir):
            match = self.frame_name_pattern.fullmatch(filename)
            if match:
                frame_count += 1
                max_index = max(max_index, int(match.group(1)))

        if frame_count == 0:
            raise ValueError(f"No frame_*.jpg images found in {frame_dir}")
        if frame_count != max_index + 1:
            raise ValueError(f"Frame files in {frame_dir} are not a continuous sequence from frame 0")
        return frame_count

    def _index_dataset(self):
        for filename in sorted(os.listdir(self.dataset_dir)):
            if not filename.startswith("tasks_sub") or not filename.endswith(".csv"):
                continue

            subject_id = filename.removeprefix("tasks_sub").removesuffix(".csv")
            if self.subject_ids is not None and subject_id not in self.subject_ids:
                continue

            csv_path = os.path.join(self.dataset_dir, filename)
            frame_dir = self._get_frame_directory(subject_id)
            if frame_dir is None:
                continue

            eye_name = os.path.basename(frame_dir).split("_")[0]
            annotation_path = os.path.join(self.dataset_dir, f"{eye_name}_sub{subject_id}.txt")
            if not os.path.isfile(annotation_path):
                raise FileNotFoundError(f"Missing frame timestamps: {annotation_path}")

            task_df = pd.read_csv(csv_path)
            if task_df.shape[1] < 2:
                raise ValueError(f"Expected task block and action columns in {csv_path}")

            timestamps = pd.read_csv(
                annotation_path,
                sep=r"\s+",
                header=None,
                skiprows=1,
                usecols=[0],
                engine="python",
            ).iloc[:, 0].to_numpy(dtype=np.float64)
            if len(timestamps) != len(task_df):
                raise ValueError(
                    f"Task rows ({len(task_df)}) do not match timestamp rows "
                    f"({len(timestamps)}) for subject {subject_id}"
                )
            if not np.isfinite(timestamps).all():
                raise ValueError(f"Non-finite frame timestamp in {annotation_path}")

            frame_indices = np.rint(timestamps * self.frame_rate).astype(np.int64)
            if (
                (frame_indices < 0).any()
                or (np.abs(timestamps * self.frame_rate - frame_indices) > 0.1).any()
            ):
                raise ValueError(f"Frame timestamps do not map to frame indices in {annotation_path}")

            frame_count = self._get_frame_count(frame_dir)
            if (frame_indices >= frame_count).any():
                raise ValueError(
                    f"Timestamp references a missing frame in {frame_dir} "
                    f"(frame count: {frame_count})"
                )

            current_segment = []
            current_key = None
            current_label = None
            for frame_idx, row in zip(
                frame_indices,
                task_df.itertuples(index=False, name=None),
            ):
                task_block = str(row[0]).strip()
                action = str(row[1]).strip()
                label = self.load_map.get(task_block)
                is_task_action = action in self.task_actions.get(task_block, set())
                if self.only_cognitive and not task_block.startswith("cog"):
                    is_task_action = False

                segment_key = (task_block, action)
                is_contiguous = not current_segment or frame_idx == current_segment[-1] + 1
                if (
                    label is None
                    or not is_task_action
                    or not is_contiguous
                    or segment_key != current_key
                ):
                    self._extract_clips_from_segment(frame_dir, current_segment, current_label)
                    current_segment = []
                    current_key = None
                    current_label = None

                if label is not None and is_task_action:
                    if not current_segment:
                        current_key = segment_key
                        current_label = label
                    current_segment.append(int(frame_idx))

            self._extract_clips_from_segment(frame_dir, current_segment, current_label)

    def _extract_clips_from_segment(self, frame_dir, frame_indices, label):
        if not frame_indices or label is None:
            return

        required_span = (self.num_frames - 1) * self.frame_stride + 1
        if len(frame_indices) < required_span:
            return

        for start_pos in range(0, len(frame_indices) - required_span + 1, self.clip_stride):
            self.samples.append((frame_dir, frame_indices[start_pos], label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame_dir, start_frame, label = self.samples[idx]
        frames = []

        for offset in range(self.num_frames):
            frame_idx = start_frame + offset * self.frame_stride
            frame_path = os.path.join(frame_dir, f"frame_{frame_idx:06d}.jpg")
            gray = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
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

        if self.rgb_format:
            normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        else:
            normalize = Normalize(mean=[0.485], std=[0.229])
        video_tensor = normalize(video_tensor).permute(1, 0, 2, 3)
        return video_tensor, torch.tensor(label, dtype=torch.long)


def get_subject_splits():
    """Return subject-level train, validation, and test splits."""
    train_subjects = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    val_subjects = [13, 14, 16]
    test_subjects = [17, 18, 19]
    return train_subjects, val_subjects, test_subjects


dataset_path = "/work/cvcs2026/LZMM/IREye4Task"
train_subs, val_subs, test_subs = get_subject_splits()

train_transforms = v2.Compose([
    v2.RandomResizedCrop(
        size=(224, 224),
        scale=(0.85, 1.0),
        ratio=(1.0, 1.0),
        antialias=True,
    ),
    v2.RandomHorizontalFlip(p=0.5),
    v2.ColorJitter(brightness=0.2, contrast=0.2),
])

val_test_transforms = v2.Compose([
    v2.Resize((224, 224), antialias=True),
])

train_dataset = IREye4TaskVideoDataset(
    dataset_dir=dataset_path,
    subject_ids=train_subs,
    num_frames=16,
    transform=train_transforms,
    clip_stride=8,
    rgb_format=True,
)

val_dataset = IREye4TaskVideoDataset(
    dataset_dir=dataset_path,
    subject_ids=val_subs,
    num_frames=16,
    transform=val_test_transforms,
    clip_stride=16,
    rgb_format=True,
)

test_dataset = IREye4TaskVideoDataset(
    dataset_dir=dataset_path,
    subject_ids=test_subs,
    num_frames=16,
    transform=val_test_transforms,
    clip_stride=16,
    rgb_format=True,
)