from cv2 import imread
from cv2 import imshow
from cv2 import waitKey
from cv2 import destroyAllWindows
from cv2 import CascadeClassifier
from cv2 import rectangle
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pafy



#classifier=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

video=cv2.VideoCapture(0)
video=cv2.VideoCapture('t.mp4')
#URL = "https://youtu.be/AY6FDIec6-Y"
#play = pafy.new(URL)
#best = play.getbest()
#video = cv2.VideoCapture(best.url) 
#camera_ip = "rtsp://username:password@IP/port"
#video = cv2.VideoCapture(camera_ip)
classifier = CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')

#print(video)

#classifier = CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')
#bboxes = classifier.detectMultiScale(video)

if(video.isOpened())==False:
	print("Error in opening video")

frame_width=int(video.get(3))
frame_height=int(video.get(4))
out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))

while(video.isOpened()):
	ret,frame = video.read()
	print(frame.shape)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	bboxes = classifier.detectMultiScale(frame,1.1, 1)
	for (x, y, w, h) in bboxes:
    		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 6)
    		cv2.imshow(' ',frame)
    		cv2.imshow('',frame)
    		cv2.waitKey(1)&0xFF== ord('a')

	#if ret==True:
		#cv2.imshow('',frame)
		#cv2.waitKey(1)&0xFF== ord('a')



	
video_cap.release()




def main():
    #classifier = CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml.xml')
    #bboxes = classifier.detectMultiScale(video)
    detect(classifier)
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()




