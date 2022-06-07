'''
resize image to 224*224, return the tensor
'''
import torchvision.transforms as transforms
import cv2 as cv
        
def preprocess_img(image):

    img=cv2.imread(image)
    process=cv2.resize(img,(224,224))
    transf = transforms.ToTensor()
    img_tensor = transf(img)
    
    return img_tensor
