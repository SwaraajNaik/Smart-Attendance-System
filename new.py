
import cv2
face_cascade = cv2.CascadeClassifier('C:\\Users\\swara\\Desktop\\Envision\\haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('C:\\Users\\swara\\Desktop\\Envision\\haarcascade_eye.xml')
cap = cv2.VideoCapture(0)
while 1:
	ret, img = cap.read()
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.2, 4)
	for (x,y,w,h) in faces:
		cv2.rectangle(img,(x,y),(x+w,y+h),(155,205,90),4)
		gray = gray[y:y+h, x:x+w]
		color = img[y:y+h, x:x+w]
		eyes = eye_cascade.detectMultiScale(gray)
		for (ex,ey,ew,eh) in eyes:
			cv2.rectangle(color,(ex,ey),(ex+ew,ey+eh),(0,17,55),2)
	cv2.imshow('img',img)
	k = cv2.waitKey(20) & 0xff
	if k == 27:
		break
cap.release()
cv2.destroyAllWindows()
