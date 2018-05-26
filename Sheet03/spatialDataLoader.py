from __future__ import division
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import torch.optim.lr_scheduler as slr
import torch.nn as nn
import torch.autograd as ag
import torch as tch
from PIL import Image
import numpy as np
import cv2
import sys
import os
import random
import shutil
import time

# Some parameters
VIDEO_FRAME_SAMPLE_RATE = 10
CONVERT = True
VIDEO_INPUT_FRAME_COUNT = 3
TRAIN_BATCH_SIZE = 2 
NWORKERS_LOADER = 4 
SHUFFLE_LOADER = True
CROP_SIZE_TF = 224 
HORIZONTAL_FLIP_TF = True 
NORM_MEANS_TF = [0.485, 0.456, 0.406] 
NORM_STDS_TF = [0.229, 0.224, 0.225]
NACTION_CLASSES = 101
NEPOCHS = 3
INITIAL_LR = 0.01
MOMENTUM_VAL = 0.9
MILESTONES_LR = [1,2]

# Some constants
VIDEO_EXTN = ".avi"
FRAME_EXTN = ".jpg"
DATA_DIR = "/home/arc/VA_Assignments/Sheet03/mini-UCF-101"
FRAMES_DIR_TRAIN = "/home/arc/VA_Assignments/Sheet03/mini-UCF-101-frames-train"
FRAMES_DIR_TEST = "/home/arc/VA_Assignments/Sheet03/mini-UCF-101-frames-test"
VIDEOLIST_TRAIN = "/home/arc/VA_Assignments/Sheet03/demoTrain.txt"
VIDEOLIST_TEST = "/home/arc/VA_Assignments/Sheet03/demoTest.txt"
ACTIONLABEL_FILE = "/home/arc/VA_Assignments/Sheet03/mini-UCF-101/ucfTrainTestlist/classInd.txt"
CHECKPOINT_DIR = "./checkpoints"

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
	torch.save(modelState, ckpLoc)
	if isBest:
		shutil.copyfile(ckpLoc, bestModel)


def getOneHot(label, nClasses):
	"""
	Get one-hot encoding for the supplied label.
	"""
	assert label < nClasses
	oh = np.zeros((1,nClasses))
	oh[0, label - 1] = 1
	return oh


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
	else:
		# The test video list does not have numeric labels
		videoLoc = line
	videoLoc = videoLoc.strip()
	actionCategory, videoName = videoLoc.split("/") # actionCategory is a string
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


class SpatialDataset(Dataset):
	"""
	Inherits from the torch Dataset. Provides N random frames 
	from each video upon each invocation.
	"""

	def __init__(self, videoListLoc, rootDir, imageTransforms = None, sampleSize = VIDEO_INPUT_FRAME_COUNT, mode = "train", actionLabelLoc = None):
		"""
		The parameters mean:
		videoListLoc : The file with the video list.
		rootDir : The directory which has all saved frames per actionCategory per video.
		transforms : A list of torch image transforms to apply to each frame.
		sampleSize : No. of frames to sample from each video (unifrom random).
		"""
		super(SpatialDataset, self).__init__()
		if not rootDir.endswith("/"):
			self.rootDir = rootDir + "/"
		else:
			self.rootDir = rootDir
		self.imageTransforms = imageTransforms
		self.sampleSize = sampleSize
		self.mode = mode
		with open(videoListLoc, "r") as videoListFile:
			self.videoList = [line for line in videoListFile]
		if mode == "test":
			if actionLabelLoc is None:
				raise ValueError("Action label dictionary required in test mode!")
			with open(actionLabelLoc, "r") as actionLabelFile:
				# self.actionLabelDict = {line.split(" ")[1] : line.split(" ")[0] for line in actionLabelFile}
				self.actionLabelDict = {}
				for line in actionLabelFile:
					val, key = line.split(" ")
					self.actionLabelDict[key] = val
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
		_, videoName, actionLabel, actionCategory, _, _ = videoInfo(self.videoList[index], self.mode)
		if self.mode == "test":
			# the test video list does not have numeric labels
			actionLabel = self.actionLabelDict[actionCategory]
		frameDir = self.rootDir + actionCategory + "/" + videoName + "/"
		# get the number of frames in the folder
		nFrames = len([frameName for frameName in os.listdir(frameDir)])
		print nFrames, self.sampleSize, random.sample(range(1, nFrames), self.sampleSize)
		frameNames = random.sample(range(1, nFrames), self.sampleSize).sort() # get random frame names
		print frameNames
		if self.imageTransforms is not None:
			loadedFrames = [self.imageTransforms(Image.open(frameDir + frame + FRAME_EXTN)) for frame in frameNames]
		else:
			loadedFrames = [Image.open(frameDir + frame + FRAME_EXTN) for frame in frameNames]
		return loadedFrames, actionLabel # the chosen frames from the video and the action label



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



def getDataLoader(dataset, batchSize = TRAIN_BATCH_SIZE, nWorkers = NWORKERS_LOADER, shuffle = SHUFFLE_LOADER):
	"""
	Return a torch dataloader with the given parameters.
	"""
	return DataLoader(
		dataset = dataset, 
		batch_size = batchSize,
		shuffle = shuffle,
		num_workers = nWorkers)




class SpatialNetwork(object):
	"""
	A wrapper for the spatial stream.
	"""

	def __init__(self, nActionClasses, nEpochs, lr, momentumVal, trainLoader, testLoader, lrMilestones, ckpLoc, resumeLoc = None):
		"""
		Initialize the model
		"""
		super(SpatialNetwork, self).__init__()
		self.nActionClasses = nActionClasses
		self.nEpochs = nEpochs
		self.lr = lr
		self.trainLoader = trainLoader
		self.testLoader = testLoader
		# get a VGG16 model pretrained with Imagenet; load it onto the graphic memory
		self.model = models.vgg16(pretrained = True).cuda() 
		self.criterion = nn.CrossEntropyLoss().cuda() # set the loss function
		# a simple SGD optimizer
		self.optimizer = tch.optim.SGD(self.model.parameters(), self.lr, momentum = momentumVal) 
		self.resumeLoc = resumeLoc
		self.startEpoch = 0 # initialize to the first epoch
		# a multi-step scheduler that changes the learning rate as per a fixed schedule
		self.scheduler = slr.MultiStepLR(self.optimizer, lrMilestones, gamma = 0.1, last_epoch = -1)
		self.highestPrecision = 0.0
		self.isBest = False
		checkAndMakeDirectories(ckpLoc) # create the checkpoint folder if it doesn't exist already
		self.ckpLoc = ckpLoc
		if tch.cuda.is_available():
			print "GPU available!"
			self.model = nn.DataParallel(self.model)


	def train(self):
		"""
		Train for an epoch.
		"""
		self.model.train() # switch to train mode
		startTime = time.time()
		for nBatch, (data, label) in enumerate(self.trainLoader):
			label = getOneHot(self.nActionClasses, label)
			label = tch.from_numpy(label)
			op = None
			for dp in data:
				ip = ag.Variable(dp)
				if op is None:
					op = self.model(ip)
				else:
					op += self.model(ip)
			loss = self.criterion(op, label)
			self.optimizer.zero_grad()
			loss.backward()
			self.optimizer.step()
		endTime = time.time()
		duration = endTime - startTime
		print "Epoch %d completed in %lf seconds" % (self.epoch, duration)


	def validate(self):
		"""
		Validate the model.
		"""
		self.model.eval() # switch to evaluation mode
		correct = 0
		total = 0
		loss = 0
		for nBatch, (data, label) in enumerate(self.testLoader):
			ohLabel = getOneHot(self.nActionClasses, label)
			ohLabel = tch.from_numpy(ohLabel)
			op = None
			for dp in data:
				ip = ag.Variable(dp)
				if op is None:
					op = self.model(ip)
				else:
					op += self.model(ip)
			loss += self.criterion(op, ohLabel) # total loss
			prediction = np.argmax(op.cpu().numpy()) # the o/p is one-hot
			if prediction == label:
				correct += 1
			total += 1
		return (correct / total), loss


	def resume(self):
		"""
		Resume training from a checkpoint, if found.
		"""
		if not (self.resumeLoc and os.path.isfile(self.resumeLoc)):
			print "No checkpoints found; starting from scratch!"
			return
		print "Resuming training from checkpoint file: %s" % (resumeLoc)
		checkpoint = tch.load(self.resumeLoc)
		self.startEpoch = checkpoint["epoch"]
		self.highestPrecision = checkpoint["highestPrecision"]
		self.model.load_state_dict(checkpoint["model"])
		self.optimizer.load_state_dict(checkpoint["optimizer"])
		self.scheduler = slr.MultiStepLR(self.optimizer, lrMilestones, gamma = 0.1, last_epoch = self.startEpoch)
		print "Loaded checkpoint!"


	def save(self):
		"""
		Save the model state for future reference.
		"""
		makeCheckpoint({
			"epoch" : self.epoch,
			"model" : self.model.state_dict(),
			"highestPrecision" : self.highestPrecision,
			"optimizer" : self.optimizer.state_dict()
			}, self.isBest, self.ckpLoc + "spatial_ckp.pth.tar", self.ckpLoc + "spatial_best.pth.tar")
		return


	def execute(self):
		"""
		Execute all epochs. Each epoch consists of training 
		on all data followed by a validation step.
		"""
		self.resume()
		for self.epoch in range(self.startEpoch, self.nEpochs):
			self.train()
			precision, loss = self.validate()
			if precision > self.highestPrecision:
				self.highestPrecision = precision
				self.isBest = True
			self.scheduler.step(loss)
			self.save() # save state
		return



def main():
	if CONVERT:
		convertVideosToFrames(DATA_DIR, FRAMES_DIR_TRAIN, VIDEOLIST_TRAIN)
		convertVideosToFrames(DATA_DIR, FRAMES_DIR_TEST, VIDEOLIST_TEST, mode = "test")
	imageTransforms = getTransforms()
	trainDataset = SpatialDataset(VIDEOLIST_TRAIN, FRAMES_DIR_TRAIN, imageTransforms)
	trainDataLoader = getDataLoader(trainDataset)
	# the same image transforms for the test data as well
	testDataset = SpatialDataset(VIDEOLIST_TEST, FRAMES_DIR_TEST, imageTransforms, mode = "test", actionLabelLoc = ACTIONLABEL_FILE)
	testDataLoader = getDataLoader(testDataset)
	net = SpatialNetwork(NACTION_CLASSES, NEPOCHS, INITIAL_LR, MOMENTUM_VAL, trainDataLoader, testDataLoader, MILESTONES_LR, CHECKPOINT_DIR)
	net.execute()


if __name__ == '__main__':
	main()