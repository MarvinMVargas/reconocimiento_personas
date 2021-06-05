import cv2 as cv
import pandas as pd
from mrcnn.config import Config
from mrcnn import model as modellib
from mrcnn import visualize
import numpy as np
import imutils
import os
import os.path as path
import tensorflow as tf
import matplotlib.pyplot as pl
import csv
import time
from datetime import date as date2
from datetime import time
from datetime import datetime

CLASS_NAMES = open("coco_labels.txt",encoding = "utf8").read().strip().split('\n')
millis = lambda: int(round(time.time() * 1000))

class SimpleConfig(Config):
    # give the configuration a recognizable name
    NAME = "coco_inference"

    # set the number of GPUs to use along with the number of images
    # per GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # number of classes (we would normally add +1 for the background
    # but the background class is *already* included in the class
    # names)
    NUM_CLASSES = len(CLASS_NAMES)+1


def analyzeImage(img):

        font = cv.FONT_HERSHEY_COMPLEX
        img = imutils.resize(img, width=512)
        draw = img.copy()
        black = np.zeros((draw.shape[0],draw.shape[1]+200,3),dtype='uint8')
        black[:draw.shape[0],:draw.shape[1]] = draw
        draw = black
        img = img.copy()
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        res = model.detect([img],verbose = 0)[0]
        nPersons = 0
        color = (0,255,0)

        for i in range(0, len(res["scores"])):
            if(res["class_ids"][i] == 1 and res["scores"][i] > 0.9 ):
                nPersons += 1
                (startY, startX, endY, endX) = res["rois"][i]
                third = int(abs(startY-endY)/3)
                face = img[startY:startY+third,startX:endX]
                face = cv.resize(face,(100,50))
                cv.imwrite(f'faces/face{nPersons}.jpg',face)
                nOfFaces = 3
                if nPersons <= nOfFaces:
                    fragment = int((draw.shape[0]-20-50*nOfFaces)/(nOfFaces+1))
                    faceCoord = (20+fragment*nPersons+50*(nPersons-1),img.shape[1]+20)
                    draw[faceCoord[0]:faceCoord[0]+face.shape[0],faceCoord[1]:faceCoord[1]+face.shape[1]] = face

                cv.rectangle(draw, (startX, startY), (endX, endY), color, 2)
        cv.putText(draw,f"Persons: {str(nPersons)}",(img.shape[1] +20,20),font,0.7,color,2)
                    
        return nPersons,draw

config = SimpleConfig()
 
# initialize the Mask R-CNN model for inference and then load the
# weights
model = modellib.MaskRCNN(mode="inference", config=config, model_dir=os.getcwd())
model.load_weights("mask_rcnn_coco.h5", by_name=True)

cap = cv.VideoCapture('people.mp4')
w = cap.get(3)
h = cap.get(4)
fourcc = cv.VideoWriter_fourcc('M', 'J', 'P', 'G')
writer = cv.VideoWriter('recPersonas.avi', fourcc, 30 , (int(w),int(h)))


while(True):
    ret,img = cap.read()
    if ret:
        nop,draw = analyzeImage(img)

        cv.imshow('img',draw)
        writer.write(draw)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
writer.release()
cv.destroyAllWindows()

