"""code for automating data uploads via rclone"""
import subprocess as sp
import os
from pathlib import Path, PurePosixPath

class Uploader:

    def __init__(self, local_project_dir, cloud_project_dir, video_framerate=30):
        self.local_project_dir = local_project_dir
        self.cloud_project_dir = cloud_project_dir
        self.framerate = video_framerate

    def convert_and_upload(self):
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
        local_video_dir = Path(self.local_project_dir) / 'Videos'
        cloud_video_dir = PurePosixPath(self.cloud_project_dir) / 'Videos'
        vid_move_cmd = ['rclone', 'move', str(local_video_dir), str(cloud_video_dir)]
        vid_move_out = sp.run(vid_move_cmd, capture_output=True, encoding='utf-8')
        if vid_move_out.stderr:
            print(f'moving videos to cloud may have failed: \n {vid_move_out.stderr}')
        copy_cmd = ['rclone', 'copy', str(self.local_project_dir), str(self.cloud_project_dir)]
        copy_out = sp.run(copy_cmd, capture_output=True, encoding='utf-8')
        if copy_out.stderr:
            print(f'copying project files to cloud may have failed: \n {copy_out.stderr}')


