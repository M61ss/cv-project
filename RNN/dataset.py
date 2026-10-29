import re
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
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
pupil_indices = list(range(20,28))
frame_rate = 60.0
frame_name_pattern = re.compile(r"frame_(\d+)\.jpg$")


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
    """Geometric mean of the axes of the ellipse fitted on the pupil landmarks."""
    points = np.stack([landmarks[:28], landmarks[28:]], axis=1)[pupil_indices].astype(np.int32)
    _, axes, _ = cv2.fitEllipse(points)
    return np.sqrt(axes[0] * axes[1])


class IREye4TaskVideoDataset(Dataset):
    """Load fixed-length labeled clips from extracted IREye4Task eye frames."""

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
        self.dataset_dir = Path(dataset_dir)
        self.subject_ids = {str(subject) for subject in subject_ids} if subject_ids is not None else None
        self.num_frames = num_frames
        self.frame_stride, self.clip_stride = frame_stride, clip_stride
        self.target_size = target_size
        self.only_cognitive = only_cognitive
        self.rgb_format = rgb_format
        self.transform = transform
        self.normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) if self.rgb_format else Normalize(mean=[0.485], std=[0.229])
        self.samples = []

        if self.num_frames <= 0 or self.frame_stride <= 0 or self.clip_stride <= 0:
            raise ValueError("num_frames, frame_stride, and clip_stride must be positive")
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

            landmarks = pd.read_csv(
                annotation_path,
                sep=r'\s+',
                skiprows=1,
                header=None
            ).drop(columns=[0, 1]).values.tolist()

            current_segment = []
            current_diameters = []
            current_key = None
            current_label = None
            for frame_idx, row, landmarks_row in zip(
                frame_indices,
                task_df.itertuples(index=False, name=None),
                landmarks
            ):
                task_block = str(row[0]).strip()
                action = str(row[1]).strip()
                label = load_map.get(task_block)
                is_task_action = action in task_actions.get(task_block, set())
                if self.only_cognitive and not task_block.startswith("cog"):
                    is_task_action = False

                segment_key = (task_block, action)
                is_contiguous = not current_segment or frame_idx == current_segment[-1] + 1

                landmarks_row = np.array(landmarks_row, dtype=np.int32)
                diameter = pupil_eq_diameter(landmarks_row)

                if (
                    label is None
                    or not is_task_action
                    or not is_contiguous
                    or segment_key != current_key
                ):
                    self._extract_clips_from_segment(frame_dir, current_segment, current_label, current_diameters)
                    current_segment = []
                    current_diameters = []
                    current_key = None
                    current_label = None

                if label is not None and is_task_action:
                    if not current_segment:
                        current_key = segment_key
                        current_label = label
                    current_segment.append(int(frame_idx))
                    current_diameters.append(diameter)

            self._extract_clips_from_segment(frame_dir, current_segment, current_label, current_diameters)

    def _extract_clips_from_segment(self, frame_dir, frame_indices, label, diameters):
        if not frame_indices or label is None:
            return

        required_span = (self.num_frames - 1) * self.frame_stride + 1
        if len(frame_indices) < required_span:
            return

        for start_pos in range(0, len(frame_indices) - required_span + 1, self.clip_stride):
            clip_diameters = diameters[start_pos:start_pos + required_span:self.frame_stride]
            dataset_row = (frame_dir, frame_indices[start_pos], label, clip_diameters)
            self.samples.append(dataset_row)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame_dir, start_frame, label, diameters = self.samples[idx]
        frames = []

        for offset in range(self.num_frames):
            frame_idx = start_frame + offset * self.frame_stride
            frame_path = frame_dir / f"frame_{frame_idx:06d}.jpg"
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

        video_tensor = self.normalize(video_tensor).permute(1, 0, 2, 3)
        return video_tensor, torch.tensor(diameters, dtype=torch.float32), torch.tensor(label, dtype=torch.int32)


def get_subject_splits():
    """Return subject-level train, validation, and test splits."""
    train_subjects = list(range(1,13))
    val_subjects = [13, 14, 16]
    test_subjects = [17, 18, 19]
    return train_subjects, val_subjects, test_subjects

print('Loading dataset...')
print('')
start_time = time.perf_counter()

config = {
    'name': 'IREye4Task',
    'path': "/work/cvcs2026/LZMM/IREye4Task",
    'num_frames': 32,
    'clip_stride': 32
}
for k, v in config.items():
    print(f"{k}: {v}")

dataset_path = config['path']
num_frames = config['num_frames']
clip_stride = config['clip_stride']
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
    num_frames=num_frames,
    transform=train_transforms,
    clip_stride=clip_stride,
    rgb_format=True,
)

val_dataset = IREye4TaskVideoDataset(
    dataset_dir=dataset_path,
    subject_ids=val_subs,
    num_frames=num_frames,
    transform=val_test_transforms,
    clip_stride=clip_stride,
    rgb_format=True,
)

test_dataset = IREye4TaskVideoDataset(
    dataset_dir=dataset_path,
    subject_ids=test_subs,
    num_frames=num_frames,
    transform=val_test_transforms,
    clip_stride=clip_stride,
    rgb_format=True,
)

end_time = time.perf_counter()
print('Dataset loaded!')
print(f'Elapsed time: {end_time - start_time:.3f} s')
print('')