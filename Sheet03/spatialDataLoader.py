from __future__ import division
import torch as tch
import torchvision as tv
import torchvision.models as models
import numpy as np
import cv2
import sys
import os

# Some parameters
VIDEO_FRAME_SAMPLE_RATE = 10

# Some utilities

def checkAndMakeDirectories(*args):
	"""
	Check if directory already exists. If not, create it.
	"""
	for arg in args:
		if not os.path.exists(arg):
			os.makedirs(arg)


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

		if (frameIdx % N) == 0:	# only every N th frame
		    frameList.append(frame)
	    frameIdx = frameIdx + 1
	    
	cap.release()
	return frameList


def convertVideosToFrames(rootDir, saveDir, videoListLoc, sampleRate = VIDEO_FRAME_SAMPLE_RATE):
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
			videoName, actionLabel = line.split(" ") # actionLabel is a number
			actionCategory, _ = videoName.split("/") # actionCategory is a string
			videoLoc = rootDir + videoName
			frameList = extractEveryNthFrame(videoLoc, sampleRate) 
			for i in range(len(frameList)):
				checkAndMakeDirectories(saveDir + actionCategory) 
				# name of the frame
				frameLoc = saveDir + actionCategory + "/" + i + "_" + actionLabel + ".jpg"
				cv2.imwrite(frameLoc, frameList[i]) # save to disk
