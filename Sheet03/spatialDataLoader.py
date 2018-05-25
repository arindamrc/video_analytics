from __future__ import division
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import sys
import os
import random

# Some parameters
VIDEO_FRAME_SAMPLE_RATE = 10

# Some constants
VIDEO_EXTN = ".avi"
FRAME_EXTN = ".jpg"
DATA_DIR = "/home/arc/VA_Assignments/Sheet03/mini-UCF-101"
FRAMES_DIR = "/home/arc/VA_Assignments/Sheet03/mini-UCF-101-frames"
VIDEOLIST_TRAIN = "/home/arc/VA_Assignments/Sheet03/demoTrain.txt"

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


# Zuerst, we need to extract the individual frames from the videos and save them to disk. 
# We will not extract all frames from a video, but every 'N' frame.

def extractEveryNthFrame(videoLoc, N):
	"""
	Get every N th frame from the video @ videoLoc.
	"""
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
	extnLen = len(VIDEO_EXTN)
	actionLabel = None
	if mode == "train":
		videoLoc, actionLabel = line.split(" ") # actionLabel is a number
	else:
		# The test video list does not have numeric labels
		videoLoc = line
	actionCategory, videoName = videoLoc.split("/") # actionCategory is a string
	videoName = videoName[:-extnLen]
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
			frameList = extractEveryNthFrame(videoLoc, sampleRate) 
			frameDir = saveDir + actionCategory + "/" + videoName # all frames of a video go in one folder
			if all(checkAndMakeDirectories(frameDir)):
				# frames written already; continue
				continue
			for i in range(len(frameList)):
				# name of the frame
				frameLoc = frameDir + "/" + str(i) + ".jpg"
				cv2.imwrite(frameLoc, frameList[i]) # save to disk
	return


class SpatialDataset(Dataset):
	"""
	Inherits from the torch Dataset. Provides N random frames 
	from each video upon each invocation.
	"""

	def __init__(self, videoListLoc, rootDir, transforms = None, frameCount = 3, mode = "train"):
		"""
		The parameters mean:
		videoListLoc : The file with the video list.
		rootDir : The directory which has all saved frames per actionCategory per video.
		transforms : A list of torch image transforms to apply to each frame.
		frameCount : No. of frames to sample from each video (unifrom random).
		"""
		super(SpatialDataset, self).__init__()
		if not rootDir.endswith("/"):
			self.rootDir = rootDir + "/"
		else:
			self.rootDir = rootDir
		self.transforms = transforms
		self.frameCount = frameCount
		self.mode = mode
		with open(videoListLoc, "r") as videoListFile:
			videoList = [line for line in videoListFile]
		if mode == "test":
			if actionLabelLoc is None:
				raise ValueError("Action label dictionary required in test mode!")
			with open(actionLabelLoc, "r") as actionLabelFile:
				self.actionLabelDict = {key:val for (val,key) in actionLabelFile}
		return

	
	def __len__(self):
		"""
		Overridden method. Return the size of the dataset.
		"""
		return len(self.videoList)
		

	def __getitem__(self, index):
		"""
		Overridden method. Return the dataset item at the given index.
		"""
		_, videoName, actionLabel, actionCategory, _, _ = videoInfo(videoList[i])
		if mode == "test":
			# the test video list does not have numeric labels
			actionLabel = self.actionLabelDict[actionCategory]
		frameDir = self.rootDir + actionCategory + "/" + videoName + "/"
		# get the number of frames in the folder
		nFrames = len([frameName for frameName in os.listdir(frameDir) if os.path.isfile(frameName)])
		frameNames = random.sample(range(1, nFrames), self.frameCount) # get random frame names
		if self.transforms is not None:
			loadedFrames = [self.transforms(Image.open(frameDir + frame + FRAME_EXTN)) for frame in frameNames]
		else:
			loadedFrames = [Image.open(frameDir + frame + FRAME_EXTN) for frame in frameNames]
		return loadedFrames, actionLabel # the chosen frames from the video and the action label


def dataLoader(dataset, transforms = None, batchSize = 256, nWorkers = 4, shuffle = True):
	"""
	Return a torch dataloader with the given parameters.
	"""
	return DataLoader(
		dataset = dataset, 
		batch_size = batchSize,
		shuffle = shuffle,
		num_workers = nWorkers)


def main():
	convertVideosToFrames(rootDir = DATA_DIR, saveDir = FRAMES_DIR, videoListLoc = VIDEOLIST_TRAIN)
	dataset = SpatialDataset(videoListLoc = VIDEOLIST_TRAIN, rootDir = FRAMES_DIR)
	dl = dataLoader(dataset)


if __name__ == '__main__':
	main()