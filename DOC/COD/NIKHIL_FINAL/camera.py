import picamera
import picamera.array
import cv2
import numpy as np

camera = picamera.PiCamera()
camera.resolution = (1920, 1088)
camera.framerate = 30

camera.start_recording('testRecording.h264')

cap=picamera.array.PiRGBArray(camera, size = (640,368))

for frame in camera.capture_continuous(cap, use_video_port=True, resize = (640,368), format="bgr"):
    img = frame.array 
    cv2.imshow("resized image", img)
    cv2.waitKey(1)
    cap.truncate(0)
