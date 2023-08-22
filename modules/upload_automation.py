"""code for automating data uploads via rclone"""
import subprocess as sp
import os
from pathlib import Path

class Uploader:

    def __init__(self, local_project_dir, cloud_project_dir, video_framerate=30):
        self.local_project_dir = local_project_dir
        self.cloud_project_dir = cloud_project_dir
        self.framerate = video_framerate

    def upload_all(self):
        self.convert_h264s_to_mp4s()
        if not self.cloud_project_dir:
            return
        self.upload_project()


    def convert_h264s_to_mp4s(self):
        local_video_dir = self.local_project_dir / 'Videos'
        h264_paths = local_video_dir.glob('*.h264')
        for h264_p in h264_paths:
            mp4_path = h264_paths.with_suffix('.mp4')
            command = ['ffmpeg', '-analyzeduration', '100M', '-probesize', '100M', '-r',
                       str(self.framerate), '-i', h264_p, '-threads', '1', '-c:v', 'copy', '-r',
                       str(self.framerate), mp4_path]
            out = sp.run(command, capture_output=True, encoding='utf-8')
            if os.path.exists(mp4_path) and (os.path.getsize(mp4_path) > os.path.getsize(h264_p)):
                os.remove(h264_p)
            else:
                print(f'failed to convert {h264_p.name}')
                print(out.stderr)
                if os.path.exists(mp4_path):
                    os.remove(mp4_path)

    def upload_project(self):


