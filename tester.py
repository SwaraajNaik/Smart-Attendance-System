import cv2
import os
import numpy as np
import faceRecognition as fr

test_img=cv2.imread('C:\\Users\\swara\\Desktop\\Envision\\FaceRecognition-master\\TestImages\\2.jpg')#test_img path
faces_detected,gray_img=fr.faceDetection(test_img)
print("faces_detected:",faces_detected)

faces,faceID=fr.labels_for_training_data('C:\\Users\\swara\\Desktop\\Envision\\FaceRecognition-master\\trainingImages')
face_recognizer=fr.train_classifier(faces,faceID)
face_recognizer.write('trainingData.yml')

c=0
name={0:"Akshya RA1711003010111",1:"Swaraaj RA1711003010118",2:"Devesh RA1711003010116",3:"Himanshu RA1711003010131",
4:"Anish RA1711003010115",6:"AbhishiktH RA1711003010128" ,7:"Mouli RA1711003010076"}
name1={0:"Actor",1:"CSE",2:"Studnt",3:"Engineering",4:"Teacher",6:"sir",7:"Garu"}
for face in faces_detected:
    c=c+1
    (x,y,w,h)=face
    roi_gray=gray_img[y:y+h,x:x+h]
    label,confidence=face_recognizer.predict(roi_gray)#predicting the label of given image
    print("confidence:",confidence)
    print("label:",label)
    fr.draw_rect(test_img,face)
    predicted_name=name[label]
    predicted_name1=name1[label]
    if(confidence>38):#If confidence more than 37 then don't print predicted face text on screen
        continue
    fr.put_text(test_img,predicted_name,x,y)
    fr.put_text(test_img,predicted_name1+" "+str(c),x,y-23)

resized_img=cv2.resize(test_img,(1000,1000))
cv2.imshow("face dEtecetion Attendance",test_img)
cv2.waitKey(0)#Waits indefinitely until a key is pressed
cv2.destroyAllWindows





