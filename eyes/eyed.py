#importing libraries
import cv2
import numpy as np
#loading haarcascade file
cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
#loading image
img = cv2.imread('goodboy.jpg')
#resizing image
img = cv2.resize(img, (500, 500))
copy = img.copy()
gray = cv2.cvtColor(copy,cv2.COLOR_BGR2GRAY)
eyes = cascade.detectMultiScale(gray,1.3,5)
for (ex,ey,ew,eh) in eyes:
    cv2.rectangle(copy,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

stack=np.hstack([img,copy])
#shoing image
cv2.imshow('Output',stack)
cv2.waitKey(0)
