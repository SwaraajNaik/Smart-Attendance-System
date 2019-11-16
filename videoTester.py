import os
import cv2
import numpy as np
import faceRecognition as fr
import random

#This module captures images via webcam and performs face recognition
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('trainingData.yml')#Load saved training data

name={0:"Akshya RA1711003010001",1:"Swaraaj RA1711003010116",2:"Devesh RA1711003010032",3:"Himanshu RA1711003010130",
4:"Anish RA1711002030102",5:"Paras RA1711003010088",6:"Abhishikth RA1711003010128",7:"mOULI Garu RA1711003010076"}
name1={0:"Actor",1:"CSE",2:"Studnt",3:"Engineering",4:"Teacher",5:"DADDY",6:"sirr",7:"sweatman"}

cap=cv2.VideoCapture(0)
c=0
while True:
    ret,test_img=cap.read()# captures frame and returns boolean value and captured image
    faces_detected,gray_img=fr.faceDetection(test_img)



    for (x,y,w,h) in faces_detected:
      l=[(0,255,100),(255,0,100),(0,100,255)]
      cv2.rectangle(test_img,(x,y),(x+w+2,y+h),random.choice(l),thickness=2)

    resized_img = cv2.resize(test_img, (800, 700))
    cv2.imshow('face detection Tutorial ',resized_img)
    cv2.waitKey(10)


    for face in faces_detected:
        (x,y,w,h)=face
        roi_gray=gray_img[y:y+w, x:x+h]
        label,confidence=face_recognizer.predict(roi_gray)#predicting the label of given image
        print("confidence:",confidence)
        print("label:",label)
        fr.draw_rect(test_img,face)
        predicted_name = name[label]
        predicted_name1=name1[label]
        if confidence < 55:#If confidence less than  then don't priqnt predicted face text on screen
           c=c+1
           fr.put_text(test_img,predicted_name,x,y)
           fr.put_text(test_img,predicted_name1+"  "+str(c),x,y-24)
           if c>2:
               fr.put_text(test_img,"Duplicate",x,y-50)
    resized_img = cv2.resize(test_img, (1000, 700))
    cv2.imshow('face recognition',resized_img)
    if cv2.waitKey(10) == ord('q'):
        break


cap.release()
cv2.destroyAllWindows

