from __future__ import division
import torch as tch
import numpy as np
import cv2
import os
import shutil

# Some utilities

def checkAndMakeDirectories(*args):
	"""
	Check if directory already exists. If not, create it.
	Returns a list of booleans. Item is true if the i th directory
	was already there.
	"""
	exists = [True] * len(args)
	for i in range(len(args)):
		arg = args[i]
		if not os.path.exists(arg):
			exists[i] = False
			os.makedirs(arg)
	return exists


def makeCheckpoint(modelState, isBest, ckpLoc, bestModel):
	"""
	Save model state to resume from later.
	"""
	tch.save(modelState, ckpLoc)
	if isBest:
		shutil.copyfile(ckpLoc, bestModel)


def getOneHot(label, nClasses):
	"""
	Get one-hot encoding for the supplied label.
	"""
	assert label < nClasses
	oh = np.zeros((1, nClasses), dtype = np.float32)
	oh[0, label - 1] = 1.0
	return tch.from_numpy(oh)


# Zuerst, we need to extract the individual frames from the videos and save them to disk. 
# We will not extract all frames from a video, but every 'N' frame.

def extractEveryNthFrame(videoLoc, N):
	"""
	Get every N th frame from the video @ videoLoc.
	"""
	if not (os.path.exists(videoLoc) and os.path.isfile(videoLoc)):
		raise ValueError("Video does not exist: %s" % (videoLoc))
	cap = cv2.VideoCapture(videoLoc)
	frameList = [] #an empty list to hold the frames
	frameIdx = 0
	while(cap.isOpened()):
		ret, frame = cap.read()
		if not ret:
			print "capture done"
			break
		if frameIdx % N == 0:
			frameList.append(frame)
		frameIdx = frameIdx + 1
	cap.release()
	return frameList



def videoInfo(line, mode):
	"""
	Extract video information from video list file's line.
	Returns information in this order:
	video location, video name, action label (number), action category (string), group number, clip number
	"""
	actionLabel = None
	if mode == "train":
		videoLoc, actionLabel = line.split(" ") # actionLabel is a number
		actionLabel = actionLabel.strip()
	else:
		# The test video list does not have numeric labels
		videoLoc = line
	videoLoc = videoLoc.strip()
	actionCategory, videoName = videoLoc.split("/") # actionCategory is a string
	actionCategory = actionCategory.strip()
	videoName = videoName[:videoName.rfind(".")] # get name without the video extension
	_, _, ngroup, nclip = videoName.split("_")
	return videoLoc, videoName, actionLabel, actionCategory, ngroup, nclip



def convertVideosToFrames(rootDir, saveDir, videoListLoc, sampleRate = VIDEO_FRAME_SAMPLE_RATE, mode = "train"):
	"""
	Read video list and convert each to a collection of sampled frames. 
	The sampling rate is given by sampleRate (default 10).
	The format of the video list file is: 
	action-category/v_action-category_group-number_clip-number.avi action-category-index
	The frames are saved under the 'saveDir'. Each action category has its own subfolder.
	The saved frames are named as frame-index_action-label.jpg.
	"""
	if not rootDir.endswith("/"):
		rootDir = rootDir + "/"
	if not saveDir.endswith("/"):
		saveDir = saveDir + "/"
	with open(videoListLoc, "r") as videoListFile:
		for line in videoListFile: # for each video
			videoLoc, videoName, _, actionCategory, _, _ = videoInfo(line, mode)
			videoLoc = rootDir + videoLoc
			frameDir = saveDir + actionCategory + "/" + videoName # all frames of a video go in one folder
			if all(checkAndMakeDirectories(frameDir)):
				# frames written already; continue
				continue
			frameList = extractEveryNthFrame(videoLoc, sampleRate) 
			for i in range(len(frameList)):
				# name of the frame
				frameLoc = frameDir + "/" + str(i) + ".jpg"
				cv2.imwrite(frameLoc, frameList[i]) # save to disk
	return



def getDataLoader(dataset, batchSize = TRAIN_BATCH_SIZE, nWorkers = NWORKERS_LOADER, shuffle = SHUFFLE_LOADER):
	"""
	Return a torch dataloader with the given parameters.
	"""
	return DataLoader(
		dataset = dataset, 
		batch_size = batchSize,
		shuffle = shuffle,
		num_workers = nWorkers)



def getTransforms(cropSize = CROP_SIZE_TF, hortizontalFlip = HORIZONTAL_FLIP_TF, normMeans = NORM_MEANS_TF, normStds = NORM_STDS_TF):
	"""
	Get image transformations based of provided parameters.
	"""
	imgTrans = []
	if cropSize:
		imgTrans.append(transforms.RandomCrop(224))
	if hortizontalFlip:
		imgTrans.append(transforms.RandomHorizontalFlip())
	imgTrans.append(transforms.ToTensor())
	if normMeans and normStds:
		imgTrans.append(transforms.Normalize(mean = normMeans, std = normStds))
	return transforms.Compose(imgTrans)
