import cv2


class MockDataCollector:

    def __init__(self, source_video, ):
        self.source_video = source_video
        self.cap = cv2.VideoCapture(self.source_video)
        self.resolution = (int(self.cap.get(3)), int(self.cap.get(4)))
        self.framerate = int(self.cap.get(cv2.CAP_PROP_FPS))

    def advance_video(self, n_frames):
        current_frame = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame + n_frames)

    def capture_frame(self):
        ret, img = self.cap.read()
        if ret:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            self.shutdown()
            return False

    def shutdown(self):
        self.cap.release()
