import cv2

class MockDataCollector:

    def __init__(self, source_video, framegrab_interval):
        self.source_video = source_video
        self.cap = cv2.VideoCapture(str(self.source_video))
        self.resolution = (int(self.cap.get(3)), int(self.cap.get(4)))
        self.framerate = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.framestep = int(self.framerate * framegrab_interval)
        self.current_frame = 1

    def capture_frame(self):
        while self.current_frame % self.framestep:
            ret, img = self.cap.read()
            self.current_frame += 1
            if not ret:
                self.cap.release()
                return False
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def shutdown(self):
        self.cap.release()
