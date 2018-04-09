import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dst
from scipy.spatial import ConvexHull

PREDICTOR_PATH = "/usr/bin/shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(PREDICTOR_PATH)
#cascade_path='haarcascade_frontalface_default.xml'
#cascade = cv2.CascadeClassifier(cascade_path)
detector = dlib.get_frontal_face_detector()


MOUTH_OUTLINE_POINTS = list(range(48,61))

def get_landmarks(im):
    rects = detector(im, 1)

    if len(rects) > 1:
        return "error"
    if len(rects) == 0:
        return "error"
    return np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])


def annotate_landmarks(im, landmarks):
    im = im.copy()
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        cv2.putText(im, str(idx), pos,
                    fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale=0.4,
                    color=(0, 0, 255))
        cv2.circle(im, pos, 3, color=(0, 255, 255))
    return im

def top_lip(landmarks):
    top_lip_pts = []
    for i in range(50,53):
        top_lip_pts.append(landmarks[i])
    for i in range(61,64):
        top_lip_pts.append(landmarks[i])
    top_lip_all_pts = np.squeeze(np.asarray(top_lip_pts))
    top_lip_mean = np.mean(top_lip_pts, axis=0)
    return int(top_lip_mean[:,1])

def bottom_lip(landmarks):
    bottom_lip_pts = []
    for i in range(65,68):
        bottom_lip_pts.append(landmarks[i])
    for i in range(56,59):
        bottom_lip_pts.append(landmarks[i])
    bottom_lip_all_pts = np.squeeze(np.asarray(bottom_lip_pts))
    bottom_lip_mean = np.mean(bottom_lip_pts, axis=0)
    return int(bottom_lip_mean[:,1])

def mouth_open(image):
    landmarks = get_landmarks(image)
    
    if landmarks == "error":
        return image, 0
    
    image_with_landmarks = annotate_landmarks(image, landmarks)
    top_lip_center = top_lip(landmarks)
    bottom_lip_center = bottom_lip(landmarks)
    lip_distance = abs(top_lip_center - bottom_lip_center)
    return image_with_landmarks, lip_distance

    #cv2.imshow('Result', image_with_landmarks)
    #cv2.imwrite('image_with_landmarks.jpg',image_with_landmarks)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

cap = cv2.VideoCapture(0)
yawns = 0
yawn_status = False 
imgTongue=cv2.imread('tongue.png',-1)
#small_res=cv2.resize(small,(0,0),fx=0.6,fy=0.6)
orig_mask = imgTongue[:,:,3]

orig_mask_inv = cv2.bitwise_not(orig_mask)

imgTongue = imgTongue[:,:,0:3]

origTongueHeight, origTongueWidth = imgTongue.shape[:2]

def lip_size(lip):
    lipWidth = dst.euclidean(lip[0],lip[6])
    hull = ConvexHull(lip)
    lipCenter = np.mean(lip[hull.vertices,:],axis=0)
    lipCenter = lipCenter.astype(int)
    return int(lipWidth), lipCenter

def place_lip(frame,lipCenter,lipSize):
    lipSize = int(lipSize*1.5)
    x1 = int(lipCenter[0,0] - (lipSize/2))
    x2 = int(lipCenter[0,0] + (lipSize/2))
    y1 = int(lipCenter[0,1] - (lipSize)/5)
    y2 = int(lipCenter[0,1] + (lipSize)/2)

    h, w = frame.shape[:2]
    if x1<0:
        x1=0
    if y1<0:
        y1=0
    if x2>w:
        x2=w
    if y2>h:
        y2=h

    lipOverlayWidth = x2-x1
    lipOverlayHeight = (y2-y1)

    lipOverlay = cv2.resize(imgTongue,(lipOverlayWidth,lipOverlayHeight),interpolation=cv2.INTER_AREA)
    mask = cv2.resize(orig_mask,(lipOverlayWidth,lipOverlayHeight),interpolation=cv2.INTER_AREA)
    mask_inv = cv2.resize(orig_mask_inv,(lipOverlayWidth,lipOverlayHeight),interpolation=cv2.INTER_AREA)

    roi = frame[y1:y2,x1:x2]

    roi_bg = cv2.bitwise_and(roi,roi,mask=mask_inv)
    roi_fg = cv2.bitwise_and(lipOverlay,lipOverlay,mask=mask)
    dst = cv2.add(roi_bg,roi_fg)
    frame[y1:y2,x1:x2] = dst

ton=cv2.resize(imgTongue,(100,50))
while True:
    ret, frame = cap.read()   
    image_landmarks, lip_distance = mouth_open(frame)
    landmarks=get_landmarks(frame)
    lip = landmarks[MOUTH_OUTLINE_POINTS]
    prev_yawn_status = yawn_status  
    lipsize,lipcenter = lip_size(lip)
    if lip_distance > 25:
        
        
        

        yawn_status = True 
        
        cv2.putText(frame, "Subject is Licking", (50,450), 
                    cv2.FONT_HERSHEY_COMPLEX, 1,(0,0,255),2)
        

        output_text = " Lick Count: " + str(yawns + 1)

        cv2.putText(frame, output_text, (50,50),
                    cv2.FONT_HERSHEY_COMPLEX, 1,(0,255,127),2)
        point=landmarks[48]
        x=point[0,0]
        y=point[0,1]
        #frame[y:y+ton.shape[0], x:x+ton.shape[1]] = ton
        place_lip(frame,lipcenter,lipsize)
        
    else:
        yawn_status = False 
         
    if prev_yawn_status == True and yawn_status == False:
        yawns += 1

    cv2.imshow('Live Landmarks', image_landmarks )
    cv2.imshow('Yawn Detection', frame )
    
    if cv2.waitKey(1) == 13: #13 is the Enter Key
        break
        
cap.release()
cv2.destroyAllWindows() 