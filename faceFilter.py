import numpy as np 
import cv2
import dlib
from scipy.spatial import distance as dst 
from scipy.spatial import ConvexHull
from PIL import Image

PREDICTOR_PATH = "/usr/bin/shape_predictor_68_face_landmarks.dat"

FULL_POINTS = list(range(0,68))
FACE_POINTS = list(range(17,68))

##########JAW(contains jaw ;) ;)###########################
JAWLINE_POINTS = list(range(0,17))

##########Face##########################
RIGHT_EYEBROW_POINTS = list(range(17,22))
LEFT_EYEBROW_POINTS = list(range(22,27))
NOSE_POINTS = list(range(27,36))
RIGHT_EYE_POINTS = list(range(36,42))
LEFT_EYE_POINTS = list(range(42,48))
MOUTH_OUTLINE_POINTS = list(range(48,61))
MOUTH_INNER_POINTS = list(range(61,68))


##############Detection and Prediction#########
detector = dlib.get_frontal_face_detector()

predictor = dlib.shape_predictor(PREDICTOR_PATH)


##################Function to calculate eye sizes###
def eye_size(eye):
	eyeWidth = dst.euclidean(eye[0],eye[3])
	hull = ConvexHull(eye)
	eyeCenter = np.mean(eye[hull.vertices,:],axis=0)
	eyeCenter = eyeCenter.astype(int)
	return int(eyeWidth),eyeCenter

def lip_size(lip):
	lipWidth = dst.euclidean(lip[0],lip[6])
	hull = ConvexHull(lip)
	lipCenter = np.mean(lip[hull.vertices,:],axis=0)
	lipCenter = lipCenter.astype(int)
	return int(lipWidth),lipCenter

def beard_size(beard):
	beardWidth = dst.euclidean(beard[2],beard[14])
	hull = ConvexHull(beard)
	beardCenter = np.mean(beard[hull.vertices,:],axis=0)
	beardCenter = beardCenter.astype(int)
	return int(beardWidth),beardCenter

def face_size(face):
	faceWidth = dst.euclidean(face[0],face[16])
	hull = ConvexHull(face)
	faceCenter = np.mean(face[hull.vertices,:],axis=0)
	faceCenter = faceCenter.astype(int)
	return int(faceWidth),faceCenter

	
##################Funtion to calculate lip sizes####

#################Function to place the overlay on to the face image###########
def place_beard(frame,beardCenter,beardSize):
	beardSize = int(beardSize * 1.5)
	x1 = int(beardCenter[0,0] - (beardSize/3))
	x2 = int(beardCenter[0,0] + (beardSize/3))
	y1 = int(beardCenter[0,1] - (beardSize/3))
	y2 = int(beardCenter[0,1] + (beardSize/3))

	h, w = frame.shape[:2]

	if x1<0:
		x1=0

	if y1<0:
		y1=0

	if x2>w:
		x2=w

	if y2>h:
		y2=h

	beardOverlayWidth = x2 - x1
	beardOverlayHeight = (y2 - y1)
	beardOverlay = cv2.resize(imgBeard,(beardOverlayWidth,beardOverlayHeight),interpolation = cv2.INTER_AREA)
	mask = cv2.resize(orig_mask_beard,(beardOverlayWidth,beardOverlayHeight),interpolation=cv2.INTER_AREA)
	mask_inv = cv2.resize(orig_mask_inv_beard,(beardOverlayWidth,beardOverlayHeight),interpolation=cv2.INTER_AREA)

	roi = frame[y1:y2,x1:x2]

	roi_bg = cv2.bitwise_and(roi,roi,mask=mask_inv)
	
	roi_fg = cv2.bitwise_and(beardOverlay,beardOverlay,mask=mask)
	
	dst = cv2.add(roi_bg,roi_fg)

	frame[y1:y2,x1:x2] = dst

def place_lip(frame,lipCenter,lipSize):
	lipSize = int(lipSize * 1.5)
	x1 = int(lipCenter[0,0] - (lipSize/2))
	x2 = int(lipCenter[0,0] + (lipSize/2))
	y1 = int(lipCenter[0,1] - (lipSize/4))
	y2 = int(lipCenter[0,1] + (lipSize/4))

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

	lipOverlay = cv2.resize(imgLip,(lipOverlayWidth,lipOverlayHeight),interpolation = cv2.INTER_AREA)
	mask = cv2.resize(orig_mask_lip,(lipOverlayWidth,lipOverlayHeight),interpolation=cv2.INTER_AREA)
	mask_inv = cv2.resize(orig_mask_inv_lip,(lipOverlayWidth,lipOverlayHeight),interpolation=cv2.INTER_AREA)


	roi = frame[y1:y2,x1:x2]

	roi_bg = cv2.bitwise_and(roi,roi,mask=mask_inv)
	
	roi_fg = cv2.bitwise_and(lipOverlay,lipOverlay,mask=mask)
	
	dst = cv2.add(roi_bg,roi_fg)

	frame[y1:y2,x1:x2] = dst
def place_eye(frame,eyeCenter,eyeSize):
	#print eyeSize
	eyeSize = int(eyeSize * 1.5)
	x1 = int(eyeCenter[0,0] - (eyeSize/2))
	x2 = int(eyeCenter[0,0] + (eyeSize/2))
	y1 = int(eyeCenter[0,1] - (eyeSize/2))
	y2 = int(eyeCenter[0,1] + (eyeSize/2))

	h, w = frame.shape[:2]

	#check for clipping
	if x1 < 0:
		x1=0
	if y1 < 0:
		y1=0
	if x2 >w:
		x2=w
	if y2 > h:
		y2=h


	#print x1,y1
	#print x2,y2
	#re-calculate the size to avoid clipping
	eyeOverlayWidth = x2 - x1
	eyeOverlayHeight = y2 - y1

	#calculate the masks for the overlay
	eyeOverlay = cv2.resize(imgEye,(eyeOverlayWidth,eyeOverlayHeight),interpolation = cv2.INTER_AREA)
	mask = cv2.resize(orig_mask,(eyeOverlayWidth,eyeOverlayHeight),interpolation=cv2.INTER_AREA)
	mask_inv = cv2.resize(orig_mask_inv,(eyeOverlayWidth,eyeOverlayHeight),interpolation=cv2.INTER_AREA)
	
	#take ROI for the overlay from background, equal to size of the overlay image
	roi = frame[y1:y2,x1:x2]

	roi_bg = cv2.bitwise_and(roi,roi,mask=mask_inv)
	
	roi_fg = cv2.bitwise_and(eyeOverlay,eyeOverlay,mask=mask)
	
	dst = cv2.add(roi_bg,roi_fg)

	frame[y1:y2,x1:x2] = dst


#def place_beard(frame,beardCenter,beardSize)

def place_face(frame,faceCenter,faceSize):
	#print eyeSize
	faceSize = int(faceSize * 1.5)
	x1 = int(faceCenter[0,0] - (faceSize/2.9))
	x2 = int(faceCenter[0,0] + (faceSize/3.0))
	y1 = int(faceCenter[0,1] - (faceSize/1.6))
	y2 = int(faceCenter[0,1] + (faceSize/3.6))

	h, w = frame.shape[:2]

	#check for clipping
	if x1 < 0:
		x1=0
	if y1 < 0:
		y1=0
	if x2 >w:
		x2=w
	if y2 > h:
		y2=h


	#print x1,y1
	#print x2,y2
	#re-calculate the size to avoid clipping
	faceOverlayWidth = x2 - x1
	faceOverlayHeight = y2 - y1

	#calculate the masks for the overlay
	faceOverlay = cv2.resize(imgFace,(faceOverlayWidth,faceOverlayHeight),interpolation = cv2.INTER_AREA)
	mask = cv2.resize(orig_mask_face,(faceOverlayWidth,faceOverlayHeight),interpolation=cv2.INTER_AREA)
	mask_inv = cv2.resize(orig_mask_inv_face,(faceOverlayWidth,faceOverlayHeight),interpolation=cv2.INTER_AREA)
	
	#take ROI for the overlay from background, equal to size of the overlay image
	roi = frame[y1:y2,x1:x2]

	roi_bg = cv2.bitwise_and(roi,roi,mask=mask_inv)
	
	roi_fg = cv2.bitwise_and(faceOverlay,faceOverlay,mask=mask)
	
	dst = cv2.add(roi_bg,roi_fg)

	frame[y1:y2,x1:x2] = dst


##################Load and pre-process filter###########

imgEye = cv2.imread("eye.png",-1)
orig_mask = imgEye[:,:,3]


orig_mask_inv = cv2.bitwise_not(orig_mask)


imgEye = imgEye[:,:,0:3]

origEyeHeight, origEyeWidth = imgEye.shape[:2]


imgLip = cv2.imread("lip.png",-1)
orig_mask_lip = imgLip[:,:,3]

orig_mask_inv_lip = cv2.bitwise_not(orig_mask_lip)

imgLip = imgLip[:,:,0:3]

lipHeight, lipWidth = imgLip.shape[:2]

imgFace = cv2.imread("face.png",-1)
orig_mask_face = imgFace[:,:,3]
orig_mask_inv_face=cv2.bitwise_not(orig_mask_face)
imgFace = imgFace[:,:,0:3]
faceHeight, faceWidth = imgFace.shape[:2]

imgBeard = cv2.imread("beard.png",-1)
orig_mask_beard = imgBeard[:,:,3]
orig_mask_inv_beard = cv2.bitwise_not(orig_mask_beard)
imgBeard = imgBeard[:,:,0:3]

##################Start capturing the WebCam#########
video_capture = cv2.VideoCapture(0)

while True:
	ret, frame = video_capture.read()
	
	if ret:
		gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
		cv2.imshow("",gray)
		rects = detector(gray,0)

		for rect in rects:
			
			x = rect.left()
			y = rect.top()
			x1 = rect.right()
			y1 = rect.bottom()

			landmarks = np.matrix([[p.x,p.y] for p in predictor(frame,rect).parts()])
			print landmarks
			
			left_eye = landmarks[LEFT_EYE_POINTS]
			right_eye = landmarks[RIGHT_EYE_POINTS]

			lip = landmarks[MOUTH_OUTLINE_POINTS]

			face = landmarks[JAWLINE_POINTS]

			beard = landmarks[JAWLINE_POINTS]

			lipSize, lipCenter = lip_size(lip)

			leftEyeSize, leftEyeCenter = eye_size(left_eye)
			rightEyeSize, rightEyeCenter = eye_size(right_eye)

			faceSize, faceCenter = face_size(face)

			beardSize, beardCenter = beard_size(beard)


			print "Left - Eye Coordinates"
			#place_eye(frame,leftEyeCenter,leftEyeSize)
			print "Right - Eye Coordinates"
			#place_eye(frame,rightEyeCenter,rightEyeSize)

			#place_lip(frame,lipCenter,lipSize)

			#place_face(frame,faceCenter,faceSize)

			place_beard(frame,beardCenter,beardSize)

		cv2.imshow("Faces with Overlay",frame)

	ch = 0xFF & cv2.waitKey(1)

	if ch == ord('q'):
		break
cv2.destroyAllWindows()














