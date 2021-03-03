import cv2
from random import randrange

#load some pre-trined data on face frontals from opencv (haar cascade algorithm)
trained_face_data=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#Choose image to detect faces
#img=cv2.imread('jen.jpg')
img=cv2.imread('friends.jpg')

#Must covert it to grayscale
grayscaled_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#Detect faces
face_coordinates=trained_face_data.detectMultiScale(grayscaled_img)



#OR USING SPECIFIC VALUES

#(x,y,w,h)=face_coordinates[0]
#cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

#print(face_coordinates)


#draw rectangle
#hardcoded
#cv2.rectangle(img,(100,66),(100+156,156+66),(0,0,255),2)
#dynamically display rectangle

#colorful frames
#for (x,y,w,h) in face_coordinates:
    #cv2.rectangle(img,(x,y),(x+w,y+h),(randrange(128,255),randrange(128,255),randrange(128,255)),5)

#single ie green frames
for (x,y,w,h) in face_coordinates:
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),5)


cv2.imshow('Face detector',img)
cv2.waitKey()

print("CODE COMPLETED")