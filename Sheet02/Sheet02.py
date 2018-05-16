from __future__ import division
import numpy as np
import scipy.ndimage as spnd
import scipy.signal as sig
import cv2
import sys
import os
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.externals import joblib
from sklearn import mixture
from scipy import linalg


# Some constants 
TRN_VIDLOC = "/home/arc/VA_Assignments/Datasets/Wiezmann" # training video directory
TST_VIDLOC = "/home/arc/VA_Assignments/Datasets/Wiezmann"# testing video directory
TRN_FLNAME = "/home/arc/VA_Assignments/Datasets/Wiezmann/train.txt" # training labels file name
TST_FLNAME = "/home/arc/VA_Assignments/Datasets/Wiezmann/test.txt" # testing labels file name
TRN_TRAJDESC  = "./trn_descriptors_pca/" # location to save generated training trajectory descriptors
TST_TRAJDESC  = "./tst_descriptors_pca/" # location to save generated testing trajectory descriptors
TRN_TRAJDESC_PCA = "./trn_descriptors_pca_reduced/" # location to save reduced training trajectory descriptors
TST_TRAJDESC_PCA = "./trn_descriptors_pca_reduced/" # location to save reduced testing trajectory descriptors

# Some parameters
CORNERNESS_SCALE = 0.001 # as given in paper
SAMPLE_STEP = 5 # pixel sample step size (W)
MEDIANBLUR_KSIZE = 3 # the kernel size for median blur
TRAJVOL_T = 15 # the length of the trajectory volume (L)
TRAJVOL_X = 32 # the width of the trajectory volume (N)
TRAJVOL_Y = 32 # the height of the trajectory volume (N)
TUBE_GRID_X = 2 # grid width of trajectory volume (n_sigma)
TUBE_GRID_Y = 2 # grid height of trajectory volume (n_sigma)
TUBE_GRID_T = 3 # grid depth of trajectory volume (n_tau)
TRAJ_MINVAR = 1.732050807568877 # minimum variance in trajectory allowed to remove static trajectories
TRAJ_MAXVAR = 30 # maximum variance to remove random trajectories
TRAJ_MAXDIS = 20 # maximim displacement allowed in trajectories
TRAJ_MAXDIS_PERCENT = 0.7 # maximum displacement between two consecutive frames as a percentage of overall displacement
EPSILON = 0.000001 # a small value to prevent division by zero
DESC_DIM = 64 # the target no. of dimensions for PCA
N_GMM_COMPONENTS = 256 # no. of GMM components 

# Dictionary fields
FIELD_LOC = "loc"
FIELD_HOG = "HoG"
FIELD_HOF = "HoF"
FIELD_MBHx = "MBHx"
FIELD_MBHy = "MBHy"
FIELD_UFLOW = "uFlow"
FIELD_VFLOW = "vFlow"
FIELD_PTS = "pts"
FIELD_SHP = "shp"
FIELD_DESC = "desc"
FIELD_HEIGHT = "ht"
FIELD_WIDTH = "wd"
FIELD_FOLLOW = "fol"

# histogram bins
BINS_HOG = [45,90,135,180,225,270,315,360]
BINS_HOF = [0,45,90,135,180,225,270,315,360]
BINS_MBH = [45,90,135,180,225,270,315,360]

# other things
NP_EXTN = ".npy"
SVM_FILE = "classifier.pkl"
GMM_FILE = "gmm.pkl"

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


def normalize(arr):
	"""
	Normalize the given numpy array. Also return the norm.
	"""
	norm = np.linalg.norm(arr)
	if norm < EPSILON:
		return arr, 0.0
	return arr/norm, norm


def readFiles(videoDir, labelFile):
	"""
	Read all video in location videoDir as
	given in labelFile
	"""
	space = " "
	samples = []
	endSlash = videoDir.endswith("/")
	extnLen = len(".avi")
	with open(labelFile, "r") as f_labels:
		for line in f_labels:
			relPath,lbl = line.split(space)
			lbl = int(lbl)
			if endSlash:
				path = videoDir + relPath
			else:
				path = videoDir + "/" + relPath
			vidLbl = relPath.replace("/", "_")
			vidLbl = vidLbl[:-extnLen] # remove the '.avi' extension
			samples.append((path, vidLbl, lbl))
	return samples


##########################################################################################


def getDenseSamples(frame, trackedPts = None):
	"""
	Get a list of pixel locations that are interesting features to track.
	The frame is sampled densely every SAMPLE_STEP pixel.
	Pixels in a SAMPLE_STEP x SAMPLE_STEP neighborhood of a tracked
	location are not chosen.
	"""
	# get corner-ness values for each pixel
	# based on the shi-tomasi method
	w, h = frame.shape
	cornerResponses = cv2.cornerMinEigenVal(frame, 3, 3)
	# calculate the threshold for corner selection
	threshold = CORNERNESS_SCALE * np.max(cornerResponses)
	# reject all pixels below the corner-ness threshold
	cornerResponses[cornerResponses < threshold] = 0
	# create an array that masks everything except every SAMPLE_STEP element
	mask = np.ravel(np.zeros_like(cornerResponses))
	mask[::SAMPLE_STEP] = 1
	mask = np.reshape(mask, cornerResponses.shape)
	if trackedPts is not None:
		for pt in trackedPts:
			tl = clip2DPoint(pt - [int(SAMPLE_STEP / 2), int(SAMPLE_STEP / 2)], [0,w], [0,h])
			br = clip2DPoint(pt + [int(SAMPLE_STEP / 2), int(SAMPLE_STEP / 2)], [0,w], [0,h])
			mask[tl[0]:br[0], tl[1]:br[1]] = 0 # don't sample in the neighborhood of tracked points
	# apply the mask; element-wise multiplication
	cornerResponses = cornerResponses * mask
	# choose those locations which are still non-zero
	xs, ys = np.where(cornerResponses > 0)
	points = zip(xs,ys)
	return points


def getDenseOpticalFlow(curFrame, nxtFrame):
	"""
	Uses Farneback's algorithm to get dense optical flow between frames.
	Returns: Two matrices that contains the x and y components of 
	the optical flow respectively.
	"""
	flow = cv2.calcOpticalFlowFarneback(curFrame, nxtFrame, 0.5, 3, 15, 3, 5, 1.2, 0)
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
			FIELD_MBHy : []
		}
	return ptInfo


def initTrajectory():
	"""
	Initialize a dictionary of trajectory information.
	"""
	trajInfo = {
		FIELD_PTS : [],
		FIELD_HOG : [],
		FIELD_HOF : [],
		FIELD_MBHx : [],
		FIELD_MBHy : [],
		FIELD_SHP : [],
		FIELD_DESC : [],
		FIELD_FOLLOW : True
	}
	return trajInfo


def initTrajectories(startPoints):
	"""
	Get the starting points of different trajectories as a list of 2d points
	and convert it into a list of trajectories.
	"""
	trajectories = []
	for pt in startPoints:
		trajInfo = initTrajectory()
		ptInfo = initTrajectoryPoint(np.array(pt))
		trajInfo[FIELD_PTS].append(ptInfo)
		trajectories.append(trajInfo)
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
	mag = np.sqrt((x * x + y * y)) # the gradient magnitude
	ang = np.arctan2(y, x) * 180 + np.pi
	ang[ang < 0] = ang[ang < 0] + 360 # the gradient angle (0 to 360 degrees)
	ang = np.mod(ang, 360)
	return mag, ang



def findIntegralHistogram(magnitudes, angles, binVals):
	"""
	Interpolate and find the contribution of each pixel 
	towards the histogram bins. Calculate the integral of
	this histogram's bins for ease of computation.
	"""
	binCount = len(binVals)
	binCount = int(binCount)
	bins = []
	binSep = abs(binVals[1] - binVals[0]) # assume uniform bin separation
	for i in range(binCount):
		iBin = angles.copy()
		iBin = 1.0 - (abs(iBin - binVals[i]) / binSep)

		if i == 0 and binVals[i] != 0: 
			# only for the first bin; all votes go to the lowest bin
			# for values lower than the lowest bin denomination.
			iBin[angles < binVals[i]] = 1.0

		# discard values not between 0 and 1
		iBin[abs(iBin) > 1.0] = 0.0
		iBin[iBin < 0.0] = 0.0

		# find histogram response from the magnitude
		iBin = iBin * magnitudes
		iBin = cv2.integral(iBin) # calculate the integral
		bins.append(iBin)
	return bins



def applyMedianBlur(uFlow, vFlow):
	"""
	Apply median blur to the matrices.
	"""
	uFlowBlurred = cv2.medianBlur(uFlow, MEDIANBLUR_KSIZE)
	vFlowBlurred = cv2.medianBlur(vFlow, MEDIANBLUR_KSIZE)
	return uFlowBlurred, vFlowBlurred



def getAllIntegralHistograms(frame, uFlow, vFlow):
	"""
	Get all integral histograms of intensity gradient, flow and flow gradients.
	"""
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


def initFrame(curFrame, nxtFrame):
	"""
	Calculate data structures based on the current frame and the next.
	These include image gradients, optical flows and their gradients.
	Everything is wrapped up in a dictionary structure for subsequent use.
	"""
	uFlow, vFlow = getDenseOpticalFlow(curFrame, nxtFrame)
	# apply median filter to optical flow
	uFlowBlurred, vFlowBlurred = applyMedianBlur(uFlow, vFlow)

	pixGradX, pixGradY = getGradients(curFrame) # get frame garadients
	uMBHX, uMBHY = getGradients(uFlow) # get x-flow gradients
	vMBHX, vMBHY = getGradients(vFlow) # get y-flow gradients

	# get integral histograms of all descriptors
	intHistHoG, intHistHoF, intHistMBHx, intHistMBHy = getAllIntegralHistograms(curFrame, uFlow, vFlow)

	frameInfo = {
		FIELD_WIDTH : curFrame.shape[1],
		FIELD_HEIGHT : curFrame.shape[0],
		FIELD_UFLOW : uFlow,
		FIELD_VFLOW : vFlow,
		FIELD_HOG : intHistHoG,
		FIELD_HOF : intHistHoF,
		FIELD_MBHx : intHistMBHx,
		FIELD_MBHy : intHistMBHy
	}

	return frameInfo


def clip2DPoint(pt, xRange, yRange):
	"""
	Clips a 2D point to given range.
	"""
	pt[0] = xRange[0] if pt[0] < xRange[0] else pt[0]
	pt[0] = xRange[1] if pt[0] > xRange[1] else pt[0]
	pt[1] = yRange[0] if pt[1] < yRange[0] else pt[1]
	pt[1] = yRange[1] if pt[1] > yRange[1] else pt[1]
	return pt


def tubeSlice(ptInfo, frameInfo):
	"""
	Calculate the image descriptors around the supplied trajectory point.
	This is done per frame per trajectory.
	"""
	pt = ptInfo[FIELD_LOC]
	# get the top-left point of the slice
	tubeTL = pt - [int(TRAJVOL_X / 2), int(TRAJVOL_Y / 2)]
	# calculate the tube's spatial dimensions
	dx = int(TRAJVOL_X / TUBE_GRID_X)
	dy = int(TRAJVOL_Y / TUBE_GRID_Y)
	for col in range(TUBE_GRID_X):
		for row in range(TUBE_GRID_Y):
			# find the four boundary points of the cell.
			# perform modulo division to ensure the cell
			# doesn't overshoot the frame boundaries.
			cellTL = clip2DPoint(tubeTL + [row * dy, col * dx], [0, frameInfo[FIELD_HEIGHT]], [0, frameInfo[FIELD_WIDTH]]).astype(int)
			cellBR = tuple(clip2DPoint(cellTL + [dy, dx], [0, frameInfo[FIELD_HEIGHT]], [0, frameInfo[FIELD_WIDTH]]).astype(int))
			cellTR = tuple(clip2DPoint(cellTL + [0, col * dx], [0, frameInfo[FIELD_HEIGHT]], [0, frameInfo[FIELD_WIDTH]]).astype(int))
			cellBL = tuple(clip2DPoint(cellTL + [row * dy, 0], [0, frameInfo[FIELD_HEIGHT]], [0, frameInfo[FIELD_WIDTH]]).astype(int))
			cellTL = tuple(cellTL)
			# for all integral histograms
			for d in [FIELD_HOG, FIELD_HOF, FIELD_MBHx, FIELD_MBHy]:
				desc = frameInfo[d]
				hist = []
				for b in desc: # for all histogram bins
					# get the bin contribution
					contrib = b[cellBR] - b[cellTR] - b[cellBL] + b[cellTL]
					hist.append(contrib) # append contributions in bin order
				ptInfo[d].append(hist) # the histogram contributions of this point for this frame
		cellTL = tubeTL
	return

 
def collateSlices(trajectory):
	"""
	Collate the histogram of last TRAJVOL_T/TUBE_GRID_T slices.
	"""
	collationSlices = trajectory[FIELD_PTS][-TUBE_GRID_T:] # take from the end
	# initialize sub-cell histograms
	for d in [FIELD_HOG, FIELD_HOF, FIELD_MBHx, FIELD_MBHy]:
		# a histogram of type d per subcell
		subcellHistograms = [[]] * (TUBE_GRID_X * TUBE_GRID_Y) 
		for pt in collationSlices: # for each slice
			allSubcellDesc = pt[d] # contains a histogram per subcell
			for i in range(len(allSubcellDesc)): # for each histogram of type d in all subcells
				desc = np.ravel(np.array(allSubcellDesc[i]))
				if len(subcellHistograms[i]) == 0:
					subcellHistograms[i].extend(desc)
				else:
					subcellHistograms[i] = desc + subcellHistograms[i]
		trajectory[d].append(subcellHistograms)



def consolidateTrajectory(trajectory):
	"""
	Collate the subcells or tubes of trajectory to get a flat descriptor.
	Perform checks to accept or reject trajectory.
	"""
	allPts = np.array([ptInfo[FIELD_LOC] for ptInfo in trajectory[FIELD_PTS]]) # get the trajectory points as a 2D array
	xVar, yVar = np.var(allPts, axis = 0)
	if xVar < TRAJ_MINVAR and yVar < TRAJ_MINVAR: 
		# reject static trajectories
		return False
	if xVar > TRAJ_MAXVAR or yVar > TRAJ_MAXVAR:
		# reject trajectories with random displacements
		return False
	diff = np.diff(allPts, axis = 0)
	xDiff = diff[:,0]
	yDiff = diff[:,1]
	# calculate trajectory segment lengths
	segLen = np.sqrt(xDiff * xDiff + yDiff * yDiff)
	# calculate overall trajectory length
	trajLen = np.sum(segLen)
	# length of the maximum segment
	maxSegLen = np.max(segLen)
	if maxSegLen > TRAJ_MAXDIS:
		# trajectory overshoots maximum displacement limit
		return False
	if maxSegLen > TRAJ_MAXDIS_PERCENT * trajLen: 
		# displacement between consecutive frames too large
		return False

	# finally calculate normalized trajectory shape descriptor
	shapeDesc = np.ravel(diff)
	shapeDesc = shapeDesc / trajLen
	trajectory[FIELD_DESC].extend(shapeDesc)

	for d in [FIELD_HOG, FIELD_HOF, FIELD_MBHx, FIELD_MBHy]:
		hist = np.ravel(np.array(trajectory[d]))
		trajectory[FIELD_DESC].extend(hist)
	# normalize the trajectory descriptor
	trajectory[FIELD_DESC], _ = normalize(trajectory[FIELD_DESC])
	return trajectory




def followTrajectories(curFrame, nxtFrame, trajectories):
	"""
	Given two consecutive frames add trajectory points that are 
	present in the next frame. Uses dense optical flow to calculate
	trajectory points.
	"""
	frameInfo = initFrame(curFrame, nxtFrame)
	uFlow = frameInfo[FIELD_UFLOW] 
	vFlow = frameInfo[FIELD_VFLOW]
	minBoundary = [0,0] 
	maxBoundary = [len(curFrame) - 1, len(curFrame[0]) - 1]
	# apply median filter to optical flow
	uFlowBlurred, vFlowBlurred = applyMedianBlur(uFlow, vFlow)

	toDelete = []
	for i in range(len(trajectories)): # for each trajectory
		trajectory = trajectories[i]
		if not trajectory[FIELD_FOLLOW]:
			# this trajectory is complete; do not follow!
			continue
		lastTrackedPt = trajectory[FIELD_PTS][-1][FIELD_LOC]
		if not all(lastTrackedPt < uFlowBlurred.shape):
			# this trajectory goes beyond the image boundary
			toDelete.append(i)
			continue
		# extract the velocity vector from the flow
		u = uFlowBlurred[lastTrackedPt[0], lastTrackedPt[1]] 
		v = vFlowBlurred[lastTrackedPt[0], lastTrackedPt[1]]
		newTrajectoryPt = np.array(lastTrackedPt) + [u,v]
		newTrajectoryPt = newTrajectoryPt.astype(int)
		if any(newTrajectoryPt > maxBoundary) or any(newTrajectoryPt < minBoundary):
			# queue this trajectory for deletion; we need complete ones!
			toDelete.append(i)
			# continue on to the next one
			continue
		# initialize the trajectory point descriptor
		ptInfo = initTrajectoryPoint(newTrajectoryPt)
		# calculate descriptors of the slice of the trajectory tube 
		tubeSlice(ptInfo, frameInfo)
		trajectory[FIELD_PTS].append(ptInfo)
		if (len(trajectory[FIELD_PTS]) % int(TRAJVOL_T / TUBE_GRID_T)) == 0: # end of sub-cell reached
			collateSlices(trajectory) # combine the histograms in the time dimension
	# finally delete invalid trajectories
	for d in toDelete[::-1]:
		del trajectories[d]




def findTrajectories(videoVol):
	"""
	Get HoG, HoF, MBHx and MBHy based trajectory descriptors from the video volume.
	Returns: A 426 dimensional descriptor vector per trajectory.
	"""
	# start with the first frame
	frame = videoVol[:,:,0]
	nxtFrame = videoVol[:,:,1]
	# get all feature points by dense sampling
	sampledPoints = getDenseSamples(frame)
	trajectories = initTrajectories(sampledPoints)
	followTrajectories(frame, nxtFrame, trajectories)
	_,_,dp = videoVol.shape
	for f in range(1, dp - 1): # Need 2 frames to calculate optical flow!
		frame = videoVol[:,:,f]
		nxtFrame = videoVol[:,:,f + 1]
		# find trajectory points in the frame
		followTrajectories(frame, nxtFrame, trajectories)
		toDelete = [] # trajectories to delete
		frameTrajPts = []
		for t in range(len(trajectories)):
			trajectory = trajectories[t]
			if not trajectory[FIELD_FOLLOW]:
				# do not follow
				continue
			# add the trajectory point in this frame
			frameTrajPts.append(trajectory[FIELD_PTS][-1][FIELD_LOC])
			trajDescriptor = None
			if len(trajectory[FIELD_PTS]) >= TRAJVOL_T: # maximum length reached
				trajDescriptor = consolidateTrajectory(trajectory)
				if trajDescriptor is False: # invalid trajectory
					toDelete.append(t)
				else:
					# trajectory complete; stop following!
					trajectory[FIELD_FOLLOW] = False
		# remove invalid trajectories; the deletion list is in ascending order
		for d in toDelete[::-1]:
			del trajectories[d]
		# find new points to follow
		sampledPoints = getDenseSamples(frame, frameTrajPts)
		trajectories.extend(initTrajectories(sampledPoints))
	# finally prune incomplete trajectories
	toDelete = []
	for t in range(len(trajectories)):
		trajectory = trajectories[t]
		if trajectory[FIELD_FOLLOW] is True:
			# incomplete: queue trajectory for deletion
			toDelete.append(t)
	# remove invalid trajectories; the deletion list is in ascending order
	for d in toDelete[::-1]:
		del trajectories[d]
	return trajectories


def reduceDimensions(trajDescriptors, maxDim = DESC_DIM):
	"""
	Use PCA to reduce the dimension of the trajectory descriptors.
	The descriptors are passed as a numpy 2D array with one descriptor per row.
	"""
	pca = PCA(n_components = maxDim)
	return pca.fit_transform(trajDescriptors)


def findGaussianMixtures(X):
	"""
	Find the parameters of the GMM based on X.
	"""
	gmm = mixture.GaussianMixture(n_components = N_GMM_COMPONENTS, covariance_type = "diag")
	gmm.fit(X)
	return gmm



def getFisherVector(trajDescriptors, gmm):
	"""
	Get a fisher vector representation of a video. 
	Based on all trajectories found in the video.
	The trajDescriptors is a 2D numpy array (T x DESC_DIM) with
	a trajectory descriptor per row. 
	"""
	T = len(trajDescriptors) # sample count 
	w = gmm.weights_ # 1 x K
	mu = gmm.means_ # K x DESC_DIM
	sigma = np.sqrt(gmm.covariances_) # K x DESC_DIM
	u = gmm.predict_proba(trajDescriptors) # T x K
	K = len(w) # component count
	gamma_T_K = u * w # T x K
	gamma_T_K = gamma_T_K / np.sum(gamma_T_K, axis = 1)[:, None]
	normedGrad_alpha = (1.0 / np.sqrt(w)) * np.sum(gamma_T_K - w, axis = 0)
	normedGrad_mu = []
	normedGrad_sigma = []
	for k in range(K):
		normedGrad_mu.append((1.0 / np.sqrt(w[k])) * np.sum(((trajDescriptors - mu[k]) / sigma[k]) * gamma_T_K[:,k][:,np.newaxis], axis = 0))
		normedGrad_sigma.append((1.0 / np.sqrt(w[k])) * np.sum((((np.square(trajDescriptors - mu[k]) / np.square(sigma[k])) - 1.0) / np.sqrt(2)) * gamma_T_K[:,k][:,np.newaxis], axis = 0))
	normedGrad_mu = np.array(normedGrad_mu)
	normedGrad_sigma = np.array(normedGrad_sigma)
	FV = [] # the fisher vector
	FV.extend(normedGrad_alpha)
	FV.extend(np.ravel(normedGrad_mu))
	FV.extend(np.ravel(normedGrad_sigma))
	FV = np.array(FV)
	FV = normalize(FV / T)[0]
	return FV



def extractAndSaveDescriptors(saveLoc, samples, overwrite = False):
	"""
	Extract descriptors from video and save to disk.
	To regenerate, set overwrite to True.
	"""
	total = len(samples)
	i = 1
	for sample in samples:
		videoLoc, videoName, _ = sample
		fName = saveLoc + videoName
		if not overwrite and os.path.exists(fName + NP_EXTN):
			# don't recalculate saved trajectories
			print "Already computed: %d out of %d" % (i, total)
			i = i + 1
			continue
		videoVol = loadVideo(videoLoc)
		trajectories = findTrajectories(videoVol)
		trajList = []
		for trajectory in trajectories:
			trajList.append(trajectory[FIELD_DESC])
		print "Saving %d out of %d" % (i, total)
		trajList = np.array(trajList)
		print trajList.shape
		np.save(fName, np.array(trajList)) # save trajectories to disk	
		i = i + 1



def loadAllTrajectories(saveLoc, samples):
	"""
	Load saved trajectory descriptors for all videos into one array.
	Returns: Combined data and the range of indices for each videos trajectories.
	"""
	data = []
	index = 0
	videoIndices = []
	for sample in samples:
		_, videoName, _ = sample
		trajDescriptors = np.load(saveLoc + videoName + NP_EXTN)
		data.append(trajDescriptors)
		videoIndices.append((index, index + len(trajDescriptors)))
		index = index + len(trajDescriptors)
	data = np.concatenate(data)
	print data.shape
	print "saved data loaded."
	return data, videoIndices


def getVideoDescriptors(samples, reducedTrajectories, videoIndices, gmm):
	"""
	Get fisher vector video descriptors from trajectory descriptors
	using the GMM.
	"""
	data = []
	labels = []
	for i in range(len(samples)):
		sample = samples[i]
		start, stop = videoIndices[i]
		_, _, label = sample
		trajDescriptors = reducedTrajectories[start:stop]
		FV = getFisherVector(trajDescriptors, gmm)
		data.append(FV)
		labels.append(label)
	data = np.array(data)
	labels = np.array(labels)
	return data, labels


def saveReducedTrajectories(reducedTrajectories, videoIndices, samples, saveLoc):
	"""
	Save PCA reduced trajectories to disk.
	"""
	for i in range(len(samples)):
		videoLoc, videoName, _ = samples[i]
		fName = saveLoc + videoName
		start, stop = videoIndices[i]
		trajDescriptors = reducedTrajectories[start:stop]
		print "Saving reduced video %d to disk" % (i)
		np.save(fName, trajDescriptors) # save trajectories to disk	


##########################################################################################


def main():
	trnDescLoc = TRN_TRAJDESC
	trnDescLoc_pca = TRN_TRAJDESC_PCA
	tstDescLoc = TST_TRAJDESC
	tstDescLoc_pca = TST_TRAJDESC_PCA
	if not os.path.exists(trnDescLoc):
		os.makedirs(trnDescLoc)
	if not os.path.exists(tstDescLoc):
		os.makedirs(tstDescLoc)

	# change the video directory and label file locations 
	trnData = readFiles(TRN_VIDLOC, TRN_FLNAME)
	tstData = readFiles(TST_VIDLOC, TST_FLNAME)

	# trnData = trnData[:3]
	# tstData = tstData[:2]
		
	extractAndSaveDescriptors(trnDescLoc, trnData)
	extractAndSaveDescriptors(tstDescLoc, tstData)

	# load all saved training trajectories
	trn_allTrajectories, trn_videoIndices = loadAllTrajectories(trnDescLoc, trnData)
	# reduce dimensions using PCA
	trn_reducedTrajectories = reduceDimensions(trn_allTrajectories)
	# save the reduced trajectories
	saveReducedTrajectories(trn_reducedTrajectories, trn_videoIndices, trnData, trnDescLoc_pca)
	# find GMM of the reduced trajectories
	gmm = findGaussianMixtures(trn_reducedTrajectories)
	# save the model
	joblib.dump(gmm, GMM_FILE)
	# get video descriptor using the GMM
	svmTrainData, svmTrainLabels = getVideoDescriptors(trnData, trn_reducedTrajectories, trn_videoIndices, gmm)

	linearClassifier = svm.LinearSVC()
	linearClassifier.fit(svmTrainData, svmTrainLabels)
	joblib.dump(linearClassifier, SVM_FILE)
	print "trained."

	# load all saved testing trajectories
	tst_allTrajectories, tst_videoIndices = loadAllTrajectories(tstDescLoc, tstData)
	# reduce dimensions using PCA
	tst_reducedTrajectories = reduceDimensions(tst_allTrajectories)
	# save the reduced trajectories
	saveReducedTrajectories(tst_reducedTrajectories, tst_videoIndices, tstData, tstDescLoc_pca)
	# get video descriptor using the GMM
	svmTestData, svmTestLabels = getVideoDescriptors(tstData, tst_reducedTrajectories, tst_videoIndices, gmm)
	# linearClassifier = joblib.load(SVM_FILE)

	print "testing..."

	preds = linearClassifier.predict(svmTestData)
	acc = 0
	for (pred, actual) in zip(preds, svmTestLabels):
		if pred == actual:
			acc = acc + 1
	print "accuracy = %f percent" % ((acc * 100.0)/len(svmTestLabels))


if __name__ == '__main__':
	main()