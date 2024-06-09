import os
import pickle

import cv2
import face_recognition

cap=cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)

imgBackground=cv2.imread('Resources/background.png')

#import mode image in the list

folderModePath="Resources/Modes"
modePathList=os.listdir(folderModePath)
imgModeList=[]
for path in modePathList:
    imgModeList.append(cv2.imread(os.path.join(folderModePath,path)))


#LOAD THE ENCODING FILE

print("loading encodeded file")
file=open("EncodeFile.p","rb")
encodeListKnownWithIds=pickle.load(file)
file.close()
encodeListKnown,studentIds=encodeListKnownWithIds
print("encode file loaded")





while True:
    success,img=cap.read()
    imgS=cv2.resize(img,(0,0),None,0.25,.25)
    imgS= cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    faceCurrFrame=face_recognition.face_locations(imgS)
    encodeCurrFrame=face_recognition.face_encodings(imgS,faceCurrFrame)



    imgBackground[162:162+480,55:55+640]=img
    imgBackground[44:44+633,808:808+414] = imgModeList[0]

    for encoFace,FaceLocation in zip(encodeCurrFrame,faceCurrFrame):
        matches=face_recognition.compare_faces(encodeListKnown,encoFace)
        faceDis=face_recognition.face_distance(encodeListKnown,encoFace)
        print("matches",matches)
        print("facedis",faceDis)
# start from 58:15

    #cv2.imshow("Web cam",img)THIS IS THE WEBCAM
    cv2.imshow("Face Attendance",imgBackground)
    cv2.waitKey(1)