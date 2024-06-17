import cv2
import numpy as np

vid_path = r'C:\Users\tucke\PycharmProjects\InternetOfFish2.0\tests\resources\sample_clip.mp4'
cap = cv2.VideoCapture(str(vid_path))
resolution = (int(cap.get(3)), int(cap.get(4)))
framerate = int(cap.get(cv2.CAP_PROP_FPS))
framestep = int(framerate * 6)

roi_slice = np.s_[115:496, 65:419]

thumbnails = []

while True:
    ret, img = cap.read()
    if not ret:
        break
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img[roi_slice]
    thumbnail = cv2.resize(img, (img.shape[1] // 10, img.shape[0] // 10))
    thumbnails.append(thumbnail)

    current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame + framestep)


