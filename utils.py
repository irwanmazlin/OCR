import numpy as np
import cv2
from imutils.object_detection import non_max_suppression
import imutils
import pytesseract


def BGR2RGB(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def imgBlur(img):
    img = cv2.GaussianBlur(img, (1,1) ,0)
    return img



def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):

    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)

    else:

        r = width / float(w)
        dim = (width, int(h * r))

    resized = cv2.resize(image, dim, interpolation = inter)
    return resized


def predictions(prob_score, geo):
	(numR, numC) = prob_score.shape[2:4]
	boxes = []
	confidence_val = []

	for y in range(0, numR):
		scoresData = prob_score[0, 0, y]
		x0 = geo[0, 0, y]
		x1 = geo[0, 1, y]
		x2 = geo[0, 2, y]
		x3 = geo[0, 3, y]
		anglesData = geo[0, 4, y]

		for i in range(0, numC):
			if scoresData[i] < 0.1:
				continue

			(offX, offY) = (i * 4.0, y * 4.0)

			angle = anglesData[i]
			cos = np.cos(angle)
			sin = np.sin(angle)

			h = x0[i] + x2[i]
			w = x1[i] + x3[i]

			endX = int(offX + (cos * x1[i]) + (sin * x2[i]))
			endY = int(offY - (sin * x1[i]) + (cos * x2[i]))
			startX = int(endX - w)
			startY = int(endY - h)

			boxes.append((startX, startY, endX, endY))
			confidence_val.append(scoresData[i])

	return (boxes, confidence_val)