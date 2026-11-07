import cv2
import numpy as np
import matplotlib

print("OpenCV :", cv2.__version__)
print("NumPy :", np.__version__)
print("Matplotlib :", matplotlib.__version__)

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Webcam inaccessible")
else:
    print("Webcam détectée !")

    ret, frame = camera.read()

    if ret:
        print("Image capturée :", frame.shape)
        cv2.imwrite("test_webcam.jpg", frame)

    camera.release()