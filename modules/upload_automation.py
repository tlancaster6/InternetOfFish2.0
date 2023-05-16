"""code for automating data uploads via rclone"""


class Uploader:

    def __init__(self, local_root, cloud_root, project_id):
        self.local_root, self.cloud_root, self.project_id = local_root, cloud_root, project_id

    def upload_all(self):
        if not self.cloud_root:
            return
        self.convert_h264s_to_mp4s()

    def convert_h264s_to_mp4s(self):
        local_video_dir = self.local_root / self.project_id / 'Videos'
        h264_paths = local_video_dir.glob('*.h264')
        for h264_p in h264_paths:
            mp4_path = h264_paths.with_suffix('.mp4')
            command = ['ffmpeg', '-analyzeduration', '100M', '-probesize', '100M', '-r',
                       str(framerate), '-i', h264_p, '-threads', '1', '-c:v', 'copy', '-r',
                       str(framerate), mp4_path]

