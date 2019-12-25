from tensorflow import keras
model2 = keras.models.load_model("mnist.h5")

from PIL import Image
import PIL.ImageOps 
import numpy as np
import cv2 as cv
import math

drawing = False
font = cv.FONT_HERSHEY_SIMPLEX


def draw_circle(event, x, y, flags, param):
	global drawing
	if event == cv.EVENT_LBUTTONDOWN:
		drawing = True
	elif (drawing is True) and (event == cv.EVENT_MOUSEMOVE):
		cv.circle(img, (x, y), 5, (0, 0, 0), -1)
	elif event == cv.EVENT_LBUTTONUP:
		drawing = False


img = np.full((512, 512, 3), 255, dtype=np.uint8)
# print(img)
cv.namedWindow('image')
cv.setMouseCallback('image', draw_circle)

while 1:
    
    cv.imshow('image', img)
    if cv.waitKey(5) & 0xFF==27:
        break
    elif cv.waitKey(5) & 0xFF==8:
        img = np.full((512, 512, 3), 255, dtype=np.uint8)
    elif cv.waitKey(5) & 0xFF==13:
        
#        img = cv.imread('digit.png')
        img2 = img.copy()
        w, h, _ = img.shape
        
        change = 0
        top, bot, left, right = 0, 0, 0, 0
        for i in range(img.shape[1]):
        	if not np.all(img[:, i, :] == 255):
        		if change == 0:
        			# img[:, i, :] = 0
        			left = i
        			change = 1
        	elif change == 1:
        		# img[:, i, :] = 0
        		right = i
        		change = 0
        		break
        
        for i in range(img2.shape[0]):
        	if not np.all(img2[i, :, :] == 255):
        		if change == 0:
        			# img[i, :, :] = 0
        			top = i
        			change = 1
        	elif change == 1:
        		# img[i, :, :] = 0
        		bot = i
        		break
        # img[:, w//2, :] = 100
        widht, height = right - left, bot - top
        W1 = w/2 - widht/2
        W2 = w/2 + widht/2
        
        H1 = h/2 - height/2
        H2 = h/2 + height/2
        
        replace = img[top:bot, left:right, :]
        img2[top+1:bot, left+1:right, :] = 255
        img2[math.ceil(H1):math.ceil(H2), math.ceil(W1):math.ceil(W2), :] = replace

        
        new_img = Image.fromarray(img2).convert('L')
        new_img.thumbnail((28, 28), Image.ANTIALIAS)
        img1 = PIL.ImageOps.invert(new_img)
        img3 =  np.array(img1)/255
        cv.putText(img, str(model2.predict_classes(img3.reshape(1, 28, 28, 1))[0]), (400, 450), font, 4, (0, 255, 255), 2, cv.LINE_AA)
        cv.putText(img, "{0:.2f}%".format(model2.predict(img3.reshape(1, 28, 28, 1)).max()*100), (400, 500), font, 1, (0, 0, 255))
        
cv.destroyAllWindows()

#for i in dir(cv):
#    if "EVENT" in i:
#        print(i)
#img.thumbnail((28, 28), Image.ANTIALIAS)
#        img1 = PIL.ImageOps.invert(img)
#        img2 =  np.array(img1)
		
