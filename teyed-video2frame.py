import os
from multiprocessing import Process
import subprocess


data_dir = '/work/cvcs2026/LZMM/TEyeD/Dikablis/VIDEOS'

fps = 25
scale = '384x288'
format = 'yuvj420p'

video_paths = os.listdir(data_dir)


def decompose(video_path):
    video_name = video_path[:-4]
    video_path = os.path.join(data_dir, video_path)
    frame_folder_path = os.path.join(data_dir, video_name)
    os.makedirs(frame_folder_path, exist_ok=True)

    print(f'Decomposing video {video_name}:')
    print('')
    print(f'- src path: {video_path}')
    print(f'- dst path: {frame_folder_path}')
    print('')

    cmd = [
        'ffmpeg',
        '-i', video_path,
        '-vf', f'fps={fps},scale={scale},format={format}',
        '-q:v', '2',
        os.path.join(frame_folder_path, 'frame_%05d.jpg')
    ]

    subprocess.run(cmd, check=True, stderr=subprocess.PIPE)

    print(f'{video_name} decomposed!')


processes = []

for video_path in video_paths:
    p = Process(target=decompose, args=[video_path])
    p.start()
    processes.append(p)

for p in processes:
    p.join()
