import cv2
import sys

print('bye')
path=sys.argv[0]
print('path= '+path)
vidcap = cv2.VideoCapture(path)  # write the filename here
success, image = vidcap.read()
count = 0
sec = 0
vidcap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
while success:
    print(image)  # image frames saved as a vector
    cv2.imwrite("video_process/LSTA-master/dataset/gtea_61/frames/S2/input/1/frame%d.jpg" % count, image)  # save frame as JPEG file
    vidcap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
    success, image = vidcap.read()
    count += 1
    sec += 0.5
