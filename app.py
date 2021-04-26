import streamlit as st
import cv2
from PIL import Image
import numpy as np
import os

def getFaceBox(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    
    #converting to necessery format
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    #getting coordinates of faces
    detections = net.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            #processing coordinates of the face
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])
            #drawing a ractangle around the face
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (255, 255, 0), int(round(frameHeight/150)), 1)
    return frameOpencvDnn, bboxes

faceProto = "weights/opencv_face_detector.pbtxt"
faceModel = "weights/opencv_face_detector_uint8.pb"

ageProto = "weights/age_deploy.prototxt"
ageModel = "weights/age_net.caffemodel"

genderProto = "weights/gender_deploy.prototxt"
genderModel = "weights/gender_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

#targets
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

# Loading networks
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)
faceNet = cv2.dnn.readNet(faceModel, faceProto)

padding = 20

def age_gender_detector(frame):
    
    frame = np.array(frame)
    frameFace, bboxes = getFaceBox(faceNet, frame)
    for bbox in bboxes:
        #getting coordinates of detected face
        face = frame[max(0,bbox[1]-padding):min(bbox[3]+padding,frame.shape[0]-1),max(0,bbox[0]-padding):min(bbox[2]+padding, frame.shape[1]-1)]
        
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        #processing to recognize age and gender
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]
        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]

        label = "{},{}".format(gender, age)
        #setting age and gender near detected image
        cv2.putText(frameFace, label, (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
    return frameFace

def main():
    #setting a title
    st.title("Age and gender recognition app")
    
    #uploading an image
    image_file = st.file_uploader("Upload image", type=['jpeg', 'png', 'jpg', 'webp'])


    if image_file is not None:
        
        image = Image.open(image_file)

        result_img = age_gender_detector(image)
        
        #displaying the processed image
        st.image(result_img, use_column_width = True)
        st.success('Found something I hope')

            
if __name__ == '__main__':
    main()
