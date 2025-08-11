import pickle
import argparse
import imutils
from warnings import warn
import cv2
import sys
import numpy as np
from scipy import signal
from skimage.metrics._structural_similarity import structural_similarity

class ImageCompare:
	
	def cal_ssim(self,img1, img2):
		K = [0.01, 0.03]
		L = 255
		kernelX = cv2.getGaussianKernel(11, 1.5)
		window = kernelX * kernelX.T
	 
		M,N = np.shape(img1)

		C1 = (K[0]*L)**2
		C2 = (K[1]*L)**2
		img1 = np.float64(img1)
		img2 = np.float64(img2)
 
		mu1 = signal.convolve2d(img1, window, 'valid')
		mu2 = signal.convolve2d(img2, window, 'valid')
	
		mu1_sq = mu1*mu1
		mu2_sq = mu2*mu2
		mu1_mu2 = mu1*mu2
	
	
		sigma1_sq = signal.convolve2d(img1*img1, window, 'valid') - mu1_sq
		sigma2_sq = signal.convolve2d(img2*img2, window, 'valid') - mu2_sq
		sigma12 = signal.convolve2d(img1*img2, window, 'valid') - mu1_mu2
   
		ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
		mssim = np.mean(ssim_map)
		return mssim,ssim_map

	def grayScale(self,image):
		grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		return grayImage
	def diffAndThresholdImage(self,diff, imageA, imageB):
		thresh = cv2.threshold(diff, 0, 255,
							   cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
		cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
								cv2.CHAIN_APPROX_SIMPLE)
		cnts = imutils.grab_contours(cnts)

		for c in cnts:
			(x, y, w, h) = cv2.boundingRect(c)
			cv2.rectangle(imageA, (x, y), (x + w, y + h), (0, 0, 255), 2)
			cv2.rectangle(imageB, (x, y), (x + w, y + h), (0, 0, 255), 2)
		return thresh
	def display(self, imageA,imageB,diff,thresh):
		cv2.imshow("Original", imageA)
		cv2.imshow("Modified", imageB)
		cv2.imshow("Diff", diff)
		cv2.imshow("Thresh ", thresh)
		cv2.waitKey(0)
		sys.stdout.close()


#object creation to class

ic = ImageCompare()

newfile = 'test.txt'

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--first", required=True,
	help="first input image")
ap.add_argument("-s", "--second", required=True,
	help="second")
args = vars(ap.parse_args())
imageA = cv2.imread(args["first"])
imageB = cv2.imread(args["second"])

grayA = ic.grayScale(imageA)
grayB = ic.grayScale(imageB)

#grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
'''img = np.array(imageB, dtype=np.uint8)'''
#grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

(score, diff) = ic.cal_ssim(grayA, grayB)
diff = (diff * 255).astype("uint8")

val1=format(score)
print("VALUE (1.0) IS ORIGINAL OR FORGERY, WHAT'S YOURS ?: {}", val1)

thresh = ic.diffAndThresholdImage(diff,imageA,imageB)

with open(newfile, "ab") as fi:
  # dump your data into the file
  pickle.dump(val1, fi)

ic.display(imageA,imageB,diff,thresh)

