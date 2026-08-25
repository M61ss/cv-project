import os
from pathlib import Path
from sys import stderr
import traceback

import torch
from torchcodec.decoders import VideoDecoder
from torchvision.io import write_jpeg


data_dir = '/work/cvcs2026/LZMM/TEyeD/Dikablis/VIDEOS'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device:', device)

video_paths = list(Path(data_dir).glob('*'))

def decompose(video_path):
    video_name = video_path.stem
    try:
        frames_folder_path = video_path.with_suffix('')
        frames_folder_path.mkdir(exist_ok=True)

        print(f'Decomposing video {video_name}:')
        print('')
        print(f'- src path: {str(video_path)}')
        print(f'- dst path: {frames_folder_path}')
        print('')

        decoder = VideoDecoder(video_path, device=device)

        for i, frame in enumerate(decoder[::5]):
            write_jpeg(frame.cpu(), os.path.join(frames_folder_path, f'frame_{i:06d}.jpg'), quality=85)

        print(f'{video_name} decomposed!')

        return True
    except Exception:
        print(f'ERROR decomposing {video_name}.')
        print(f'{video_name}:', file=stderr)
        traceback.print_exc(file=stderr)

        return False
        

if __name__ == '__main__':
    not_decomposed = []
    success_count = 0

    for video_path in video_paths:
        if decompose(video_path) == True:
            success_count += 1
        else:
            not_decomposed.append(str(video_path.stem))

    print(f'Successfully decomposed {success_count} videos on {len(video_paths)}!')
    print(f'Not decomposed videos:', not_decomposed)
