'''
resize image to 224*224
'''

import cv2

def preprocess(image):

    img=cv2.imread(image)
    process=cv2.resize(img,(224,224))
    
    return process

