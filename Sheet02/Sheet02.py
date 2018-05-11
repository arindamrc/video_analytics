from __future__ import division
import numpy as np
import scipy.ndimage as spnd
import scipy.signal as sig
import cv2
import sys
import os
import sklearn.cluster as skl
from sklearn import svm
from sklearn.externals import joblib


# Some constants and parameters

TRN_VIDLOC = "/home/arc/VA_Assignments/Datasets/Wiezmann"
TST_VIDLOC = "/home/arc/VA_Assignments/Datasets/Wiezmann"
TRN_FLNAME = "/home/arc/VA_Assignments/Datasets/Wiezmann/train.txt"
TST_FLNAME = "/home/arc/VA_Assignments/Datasets/Wiezmann/test.txt"
TRN_VIDDESC  = "./trn_descriptors/"
TST_VIDDESC  = "./tst_descriptors/"
CORNERNESS_SCALE = 0.001
SAMPLE_STEP = 5
MEDIANBLUR_KSIZE = 3
TRACKLEN = 15

# Some utilities

def loadVideo(location):
	"""
	Load the video as a 3D numpy array. 
	Convert frames to gray first.
	"""
	cap = cv2.VideoCapture(location)

	frameList = [] #an empty list to hold the frames

	while(cap.isOpened()):
	    ret, frame = cap.read()
	    if not ret:
			print "capture done"
			break	

	    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # get only intensity values
	    frameList.append(frame)
	    
	cap.release()
	frameList = np.array(frameList)
	videoVol = np.stack(frameList, axis = 2) # stack the frames to get a video volume
	return videoVol


def displayVideo(videoVol, videoName = "Video", wait = 10):
	"""
	Display the 3d numpy array videoVol as a rapid sequence of still frames.
	"""
	_, _, dp = videoVol.shape
	cv2.namedWindow(videoName)
	for f in range(dp):
		frame = videoVol[:,:,f]
		cv2.imshow(videoName, frame)
		cv2.waitKey(wait)
	cv2.destroyAllWindows()


def readFiles(videoDir, labelFile):
	"""
	Read all video in location videoDir as
	given in labelFile
	"""
	space = " "
	samples = []
	end_slash = videoDir.endswith("/")
	len_extn = len(".avi")
	with open(labelFile, "r") as f_labels:
		for line in f_labels:
			relPath,lbl = line.split(space)
			lbl = int(lbl)
			if end_slash:
				path = videoDir + relPath
			else:
				path = videoDir + "/" + relPath
			vid_lbl = relPath.replace("/", "_")
			vid_lbl = vid_lbl[:-len_extn] # remove the '.avi' extension
			samples.append((path, vid_lbl, lbl))
	return samples

##########################################################################################

def getDenseSamples(frame):
	"""
	Get a list of pixel locations that are interesting features to track.
	The frame is sampled densely every SAMPLE_STEP pixel.
	"""
	# get corner-ness values for each pixel
	# based on the shi-tomasi method
	cornerResponses = cv2.cornerMinEigenVal(frame, 3, 3)
	# calculate the threshold for corner selection
	threshold = CORNERNESS_SCALE * np.max(cornerResponses)
	# reject all pixels below the corner-ness threshold
	cornerResponses[cornerResponses < threshold] = 0
	# create an array that masks everything except every SAMPLE_STEP element
	mask = np.ravel(np.zeros_like(cornerResponses))
	mask[::SAMPLE_STEP] = 1
	mask = np.reshape(mask, cornerResponses.shape)
	# apply the mask; element-wise multiplication
	cornerResponses = cornerResponses * mask
	# choose those locations which are still non-zero
	xs, ys = np.where(cornerResponses > 0)
	points = [(x,y) in zip(xs,ys)]
	return points


def getDenseOpticalFlow(curFrame, nxtframe):
	"""
	Uses Farnebacks algorithm to get dense optical flow between frames.
	Returns: Two matrices that contains the x and y components of 
	the optical flow respectively.
	"""
	flow = cv2.calcOpticalFlowFarneback(curFrame, nxtframe, 0.5, 3, 15, 3, 5, 1.2, 0)
	return flow[...,0], flow[...,1]



def findDescriptors(videoVol):
	"""
	Get HoG, HoF, MBHx and MBHy based descriptors from the video volume.
	Returns: A 426 dimensional video descriptor vector.
	"""
	_,_,dp = videoVol.shape
	for f in range(0, dp-1): # Need 2 frames to calculate optical flow!
		frame = videoVol[:,:,f]
		sampledPoints = getDenseSamples(frame)
		nxtFrame = videoVol[:,:,f+1]
		uFlow, vFlow = getDenseOpticalFlow(frame, nxtframe)
		# apply median filter to optical flow
		uFlowBlurred = cv2.medianBlur(uFlow, MEDIANBLUR_KSIZE)
		vFlowBlurred = cv2.medianBlur(vFlow, MEDIANBLUR_KSIZE)


def extractAndSaveDescriptors(saveLoc, samples):
	"""
	Extract descriptors from video and save to disk.
	"""
	for sample in samples:
		videoLoc, videoName, _ = sample
		videoVol = loadVideo(videoLoc)
		stips, videoDesc = findDescriptors(videoVol)
		np.save(saveLoc + videoName, videoDesc) # save video descriptor to disk	

##########################################################################################


def main():
	trnDescLoc = TRN_VIDDESC
	tstDescLoc = TST_VIDDESC
	if not os.path.exists(trnDescLoc):
		os.makedirs(trnDescLoc)
	if not os.path.exists(tstDescLoc):
		os.makedirs(tstDescLoc)

	trnData = readFiles(TRN_VIDLOC, TRN_FLNAME)
	tstData = readFiles(TST_VIDLOC, TST_FLNAME)
		
	extractAndSaveDescriptors(trnData)
	extractAndSaveDescriptors(tstData)



if __name__ == '__main__':
	main()