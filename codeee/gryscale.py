# open-cv library is installed as cv2 in python
# import cv2 library into this program
import cv2

# read an image using imread() function of cv2
# we have to  pass only the path of the image
img = cv2.imread(r'C:/Users/user/Desktop/pic1.jpg')

# displaying the image using imshow() function of cv2
# In this : 1st argument is name of the frame
# 2nd argument is the image matrix
cv2.imshow('original image',img)

# converting the colourfull image into grayscale image
# using cv2.COLOR_BGR2GRAY argument of
# the cvtColor() function of cv2
# in this :
# ist argument is the image matrix
# 2nd argument is the attribute
gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# displaying the gray scale image
cv2.imshow('Gray scale image',gray_img)
