"""code for automating data uploads via rclone"""
from glob import glob

class Uploader:

    def __init__(self, local_root, cloud_root, project_id):
        self.local_root, self.cloud_root, self.project_id = local_root, cloud_root, project_id

    def upload_all(self):
        if not self.cloud_root:
            return
        self.convert_h264s_to_mp4s()

    def convert_h264s_to_mp4s(self):
        local_video_dir = self.local_root / self.project_id / 'Videos'
        video_paths = glob()

