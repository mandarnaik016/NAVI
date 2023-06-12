import cv2 as cv 
import numpy as np
import pyttsx3
#import picamera
#import picamera.array

# Distance constants 
KNOWN_DISTANCE = 45 #INCHES
PERSON_WIDTH = 16 #INCHES
MOBILE_WIDTH = 3.0 #INCHES
BOTTLE_WIDTH = 3.0#INCHES

# Object detector constant 
CONFIDENCE_THRESHOLD = 0.4
NMS_THRESHOLD = 0.3

# colors for object detected
COLORS = [(255,0,0),(255,0,255),(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]
GREEN =(0,255,0)
BLACK =(0,0,0)
# defining fonts 
FONTS = cv.FONT_HERSHEY_COMPLEX

#initialization of camera
#camera = picamera.PiCamera()
#camera.resolution = (1920, 1088)
#camera.framerate = 30
#camera.start_preview()
#raw_capture = picamera.array.PiRGBArray(camera)

# getting class names from classes.txt file 
class_names = []
with open(r"/home/navigator/Downloads/classes.txt") as f:
    class_names = [cname.strip() for cname in f.readlines()]
#  setttng up opencv net
yoloNet = cv.dnn.readNet(r"/home/navigator/Downloads/yolov4-tiny.weights", r"/home/navigator/Downloads/yolov4-tiny.cfg")

yoloNet.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
yoloNet.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)

model = cv.dnn_DetectionModel(yoloNet)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)

# object detector funciton /method
def object_detector(image):
    classes, scores, boxes = model.detect(image, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    # creating empty list to add objects data
    data_list =[]
    for (classid, score, box) in zip(classes, scores, boxes):
        # define color of each, object based on its class id 
        color= COLORS[int(classid) % len(COLORS)]
    
        label = "%s : %f" % (class_names[classid], score)

        # draw rectangle on and label on object
        cv.rectangle(image, box, color, 2)
        cv.putText(image, label, (box[0], box[1]-14), FONTS, 0.5, color, 2)
    
        # getting the data 
        # 1: class name  2: object width in pixels, 3: position where have to draw text(distance)
        if classid ==0: # person class id 
            data_list.append([class_names[classid], box[2], (box[0], box[1]-2)])
        elif classid ==67:#for cell phone
            data_list.append([class_names[classid], box[2], (box[0], box[1]-2)])
        elif classid == 39:#for bottle
            data_list.append([class_names[classid], box[2], (box[0], box[1]-2)])
        # if you want inclulde more classes then you have to simply add more [elif] statements here
        # returning list containing the object data.
    return data_list



def focal_length_finder (measured_distance, real_width, width_in_rf):
    focal_length = (width_in_rf * measured_distance) / real_width

    return focal_length

# distance finder function 
def distance_finder(focal_length, real_object_width, width_in_frmae):
    distance = (real_object_width * focal_length) / width_in_frmae
    return distance


# reading the reference image from dir 
ref_person = cv.imread(r"/home/navigator/Downloads/image14.png")
ref_mobile = cv.imread(r"/home/navigator/Downloads/image4.png")
ref_bottle = cv.imread(r"/home/navigator/Downloads/Bottle_ref.jpg")

bottle_data = object_detector(ref_bottle)
bottle_width_in_rf = bottle_data[0][1]

mobile_data = object_detector(ref_mobile)
mobile_width_in_rf = mobile_data[0][1]

person_data = object_detector(ref_person)
person_width_in_rf = person_data[0][1]

print(f"Person width in pixels : {person_width_in_rf} mobile width in pixel: {mobile_width_in_rf} bottle width in pixel : {bottle_width_in_rf}")

# finding focal length 
focal_person = focal_length_finder(KNOWN_DISTANCE, PERSON_WIDTH, person_width_in_rf)

focal_mobile = focal_length_finder(KNOWN_DISTANCE, MOBILE_WIDTH, mobile_width_in_rf)

focal_bottle = focal_length_finder(KNOWN_DISTANCE, BOTTLE_WIDTH, bottle_width_in_rf)
cap = cv.VideoCapture(0)
while True:
    ret,img = cap.read()
    img = cv.resize(img,(680,460))
    data = object_detector(img) 
    # initialisation
    engine = pyttsx3.init() 
    for d in data:
        if d[0] =='person':
            distance = distance_finder(focal_person, PERSON_WIDTH, d[1])
            x, y = d[2]
            engine.say("There is"+str(d[0])+'at'+str(round(distance,2))+'inch')
            engine.runAndWait()
        elif d[0] =='cell phone':
            distance = distance_finder (focal_mobile, MOBILE_WIDTH, d[1])
            x, y = d[2]
            engine.say("There is"+str(d[0])+'at'+str(round(distance,2))+'inch')
            engine.runAndWait()
        elif d[0] =='bottle':
            distance = distance_finder (focal_bottle,BOTTLE_WIDTH, d[1])
            x, y = d[2]
            engine.say("There is"+str(d[0])+'at'+str(round(distance,2))+'inch')
            engine.runAndWait()
    #cv.rectangle(img, (x, y-3), (x+150, y+23),BLACK,-1 )
    #cv.putText(img, f'Dis: {round(distance,2)} inch', (x+5,y+13), FONTS, 0.48, GREEN, 2)

    #cv.imshow('frame',img)
    cv.waitKey(1)
    #cap.truncate(0)
        
    
    if cv.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv.destroyAllWindows()
