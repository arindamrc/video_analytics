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

FIELD_LOC = "loc"
FIELD_HOG = "HoG"
FIELD_HOF = "HoF"
FIELD_MBHx = "MBHx"
FIELD_MBHy = "MBHy"

# histogram bins
BINS_HOG = [45,90,135,180,225,270,315,360]
BINS_HOF = [0,45,90,135,180,225,270,315,360]
BINS_MBH = [45,90,135,180,225,270,315,360]


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


def initTrajectoryPoint(pt):
	"""
	Initialize a dictionary of trajectory point information.
	"""
	ptInfo = {
			FIELD_LOC : pt,
			FIELD_HOG : [],
			FIELD_HOF : [],
			FIELD_MBHx : [],
			FIELD_MBHy : [],
		}
	return ptInfo


def initTrajectories(startPoints):
	"""
	Get the starting points of different trajectories as a list of 2d points
	and convert it into a list of trajectories.
	"""
	trajectories = []
	for pt in startPoints:
		trajectory = []
		ptInfo = initTrajectoryPoint(np.array(pt))
		trajectory.append(ptInfo)
		trajectories.append(trajectory)
	return trajectories


def getGradients(frame):
	"""
	Calculates the gradient matrices(dx,dy) for this frame.
	"""
	gradX = cv2.Sobel(frame, -1, 1, 0)
	gradY = cv2.Sobel(frame, -1, 0, 1)
	return gradX, gradY


def getMagnitudeAndAngle(x, y):
	"""
	Get the magnitude and angle of vectors from their x and y components.
	"""
	mag, ang = cv2.cartToPolar(x,y)
	return mag, ang


def findIntegralHistogram(magnitudes, angles, binVals):
	"""
	Interpolate and find the contribution of each pixel 
	towards the histogram bins. Calculate the integral of
	this histogram bins for ease of computation.
	"""
	binCount = len(binVals)
	if 360 % bin_sep != 0: # ensure uniform bins
		raise ValueError('could not divide bins evenly for bin separation %f' % (bin_sep))
	binCount = int(binCount)
	bins = []
	for i in range(binCount - 1):
		bin_i = angles.copy()
		binVal = binVals[i]
		bin_i = 1.0 - (abs(bin_i - binVals[i]) / abs(binVals[i + 1] - binVals[i]))

		# discard values not between 0 and 1
		bin_i[abs(bin_i) > 1.0] = 0.0
		bin_i[bin_i < 0.0] = 0.0

		if i == 0: 
			# only for the first bin; all votes go to the lowest bin
			# for values lower than the lowest bin denomination.
			bin_i[bin_i < binVals[i]] = 1.0

		# find histogram response from the magnitude
		bin_i = bin_i * magnitudes
		bin_i = cv2.integral(bin_i) # calculate the integral
		bins.append(bin_i)
	return bins


def applyMedianBlur(uFlow, vFlow):
	"""
	Apply median blur to the matrices.
	"""
	uFlowBlurred = cv2.medianBlur(uFlow, MEDIANBLUR_KSIZE)
	vFlowBlurred = cv2.medianBlur(vFlow, MEDIANBLUR_KSIZE)
	return uFlowBlurred, vFlowBlurred

def getAllIntegralHistograms(frame, uFlow, vFlow):
	pixGradX, pixGradY = getGradients(frame)
	uMBHX, uMBHY = getGradients(uFlow)
	vMBHX, vMBHY = getGradients(vFlow)

	pixMag, pixAng = getMagnitudeAndAngle(pixGradX, pixGradY)
	MBHxMag, MBHxAng = getMagnitudeAndAngle(uMBHX, uMBHY)
	MBHyMag, MBHyAng = getMagnitudeAndAngle(vMBHX, vMBHY)
	flowMag, flowAng = getMagnitudeAndAngle(uFlow, vFlow)

	intHistHoG = findIntegralHistogram(pixMag, pixAng, BINS_HOG)
	intHistHoF = findIntegralHistogram(flowMag, flowAng, BINS_HOF)
	intHistMBHx = findIntegralHistogram(MBHxMag, MBHxAng, BINS_MBH)
	intHistMBHy = findIntegralHistogram(MBHyMag, MBHyAng, BINS_MBH)

	return intHistHoG, intHistHoF, intHistMBHx, intHistMBHy


def findTrajectories(curFrame, nxtFrame, trajectories):
	"""
	Given two consecutive frames add trajectory points that are 
	present in the next frame. Uses dense optical flow to calculate
	trajectory points.
	"""
	uFlow, vFlow = getDenseOpticalFlow(curFrame, nxtFrame)
	minBoundary = [0,0] 
	maxBoundary = [len(curFrame), len(curFrame[0]) - 1]
	# apply median filter to optical flow
	uFlowBlurred, vFlowBlurred = applyMedianBlur(uFlow, vFlow)

	pixGradX, pixGradY = getGradients(frame) # get frame garadients
	uMBHX, uMBHY = getGradients(uFlow) # get x-flow gradients
	vMBHX, vMBHY = getGradients(vFlow) # get y-flow gradients

	# get integral histograms of all descriptors
	intHistHoG, intHistHoF, intHistMBHx, intHistMBHy = getAllIntegralHistograms(frame, uFlow, vFlow)

	toDelete = []
	for i in range(len(trajectories)): # for each trajectory
		trajectory = trajectories[i]
		lastTrackedPt = trajectory[len(trajectory) - 1][FIELD_LOC]
		# extract the velocity vector from the flow
		u = uFlowBlurred[lastTrackedPt[0], lastTrackedPt[1]] 
		v = vFlowBlurred[lastTrackedPt[0], lastTrackedPt[1]]
		newTrajectoryPt = np.array(lastTrackedPt) + [u,v]
		if any(newTrajectoryPt > maxBoundary) or any(newTrajectoryPt < minBoundary):
			# remove this trajectory; we need complet ones!
			toDelete.append(i)
			# continue on to the next one
			continue
		trajectory.append(initTrajectoryPoint(newTrajectoryPt))
	for d in toDelete:
		del trajectories[i]


def findDescriptors(videoVol):
	"""
	Get HoG, HoF, MBHx and MBHy based descriptors from the video volume.
	Returns: A 426 dimensional video descriptor vector.
	"""
	# start with the first frame
	frame = videoVol[:,:,0]
	nxtframe = videoVol[:,:,1]
	# get all feature points by dense sampling
	sampledPoints = getDenseSamples(prvFrame)
	trajectories = initTrajectories(sampledPoints)
	findTrajectories(frame, nxtFrame, trajectories)
	_,_,dp = videoVol.shape
	for f in range(1, dp - 1): # Need 2 frames to calculate optical flow!
		frame = videoVol[:,:,f]
		nxtFrame = videoVol[:,:,f + 1]
		# find trajectory points in the frame
		findTrajectories(frame, nxtframe, trajectories)


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