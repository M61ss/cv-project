import os
from pathlib import Path
from multiprocessing import Pool, cpu_count
import subprocess


data_dir = '/work/cvcs2026/LZMM/TEyeD/Dikablis/ANNOTATIONS/'

threads = '1'
fps = 25
scale = '384x288'
format = 'yuvj420p'

video_paths = list(Path(data_dir).glob('*pupil_seg_2D.mp4'))

def decompose(video_path):
    video_name = video_path.stem
    frame_folder_path = video_path.with_suffix('')
    frame_folder_path.mkdir(exist_ok=True)

    print(f'Decomposing video {video_name}:')
    print('')
    print(f'- src path: {str(video_path)}')
    print(f'- dst path: {frame_folder_path}')
    print('')

    cmd = [
        'ffmpeg',
        '-y', '-threads', threads,
        '-i', str(video_path),
        '-vf', f'fps={fps},scale={scale},format={format}',
        '-q:v', '2',
        os.path.join(frame_folder_path, 'frame_%05d.jpg')
    ]

    result = subprocess.run(cmd, check=True, stderr=subprocess.PIPE)
    ok = result.returncode == 0
    err_msg = result.stderr.decode(errors='ignore') if not ok else ''
    return video_name, ok, err_msg


if __name__ == '__main__':
    n_workers = min(cpu_count(), len(video_paths)) or 1

    with Pool(processes=n_workers) as pool:
        results = pool.map(decompose, video_paths)

    success_number = 0
    for video_name, ok, err_msg in results:
        if ok:
            success_number += 1
            print(f'{video_name} decomposed!')
        else:
            print(f'FAILED: {video_name}')
            print(err_msg)

    print(f'Successfully decomposed {success_number} on {len(video_paths)}')
