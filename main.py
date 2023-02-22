import numpy as np
import cv2
from imutils.object_detection import non_max_suppression
import imutils
import pytesseract
from matplotlib import pyplot as plt
from utils import *


width = 320
height = 320

image = cv2.imread("/home/irwan/Desktop/tf2/Img/13.jpg") # set image directory
image = BGR2RGB(image)
# image = gray(image)

# image = threshhold(image)
# # image = dilation(image)
# image = thresholdinv(image)
image = image_resize(image, width = 320, height = 50)
image = imgBlur(image)

cv2.imwrite('/home/irwan/Desktop/tf2/OCR_EAST/image.jpg', image)

image = cv2.imread('/home/irwan/Desktop/tf2/OCR_EAST/image.jpg')

orig = image.copy()
(origH, origW) = image.shape[:2]

(newW, newH) = (width, height)


rW = origW / float(newW)
rH = origH / float(newH)


image = cv2.resize(image, (newW, newH))


(H, W) = image.shape[:2]

blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
	(123.68, 116.78, 103.94), swapRB=True, crop=False)


net = cv2.dnn.readNet("/home/irwan/Desktop/tf2/OCR_EAST/model/east_text_detection.pb")

layerNames = [
	"feature_fusion/Conv_7/Sigmoid",
	"feature_fusion/concat_3"]

net.setInput(blob)
(scores, geometry) = net.forward(layerNames)

(boxes, confidence_val) = predictions(scores, geometry)
boxes = non_max_suppression(np.array(boxes), probs=confidence_val)

results = []

for (startX, startY, endX, endY) in boxes:
	startX = int(startX * rW)
	startY = int(startY * rH)
	endX = int(endX * rW)
	endY = int(endY * rH)

	r = orig[startY:endY, startX:endX]

	configuration = ("-l eng --oem 1 --psm 8")
	text = pytesseract.image_to_string(r, config=configuration)

	results.append(((startX, startY, endX, endY), text))

orig_image = orig.copy()

for ((start_X, start_Y, end_X, end_Y), text) in results:
	print("{}\n".format(text))

	text = "".join([x if ord(x) < 128 else "" for x in text]).strip()
	cv2.rectangle(orig_image, (start_X, start_Y), (end_X, end_Y),
		(0, 0, 255), 2)
	cv2.putText(orig_image, text, (start_X, start_Y - 30),
		cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,0, 255), 2)



cv2.imwrite('/home/irwan/Desktop/tf2/OCR_EAST/output.jpg', orig_image)
