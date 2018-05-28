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
import torch

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
INITIAL_LR = 0.001
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
CHECKPOINT_DIR = "./checkpoints/"
SPATIAL_CKP_FILE = "spatial_ckp.pth.tar"
SPATIAL_BEST_FILE = "spatial_best.pth.tar"

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


class SpatialDataset(Dataset):
	"""
	Inherits from the torch Dataset. Provides N random frames 
	from each video upon each invocation.
	"""

	def __init__(self, videoListLoc, rootDir, imageTransforms = None, frameSampleSize = VIDEO_INPUT_FRAME_COUNT, mode = "train", actionLabelLoc = None):
		"""
		The parameters mean:
		videoListLoc : The file with the video list.
		rootDir : The directory which has all saved frames per actionCategory per video.
		transforms : A list of torch image transforms to apply to each frame.
		frameSampleSize : No. of frames to sample from each video (unifrom random).
		"""
		super(SpatialDataset, self).__init__()
		if not rootDir.endswith("/"):
			self.rootDir = rootDir + "/"
		else:
			self.rootDir = rootDir
		self.imageTransforms = imageTransforms
		self.frameSampleSize = frameSampleSize
		self.mode = mode
		with open(videoListLoc, "r") as videoListFile:
			self.videoList = [line for line in videoListFile]
		# if mode == "test":
		if actionLabelLoc is None:
			raise ValueError("Action label dictionary required!")
		with open(actionLabelLoc, "r") as actionLabelFile:
			# self.actionLabelDict = {line.split(" ")[1] : line.split(" ")[0] for line in actionLabelFile}
			self.actionLabelDict = {}
			for line in actionLabelFile:
				val, key = line.split(" ")
				key = key.strip()
				val = int(val)
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
		# frameNames = sorted(random.sample(range(1, nFrames), self.frameSampleSize)) # get random frame names
		# if self.imageTransforms is not None:
		# 	loadedFrames = [self.imageTransforms(Image.open(frameDir + str(frame) + FRAME_EXTN)) for frame in frameNames]
		# else:
		# 	loadedFrames = [transforms.ToTensor(Image.open(frameDir + str(frame) + FRAME_EXTN)) for frame in frameNames]
		# loadedFrames = tch.stack(loadedFrames) # convert the list into a tensor
		frameName = random.randint(0, nFrames - 1) # load a random frame
		if self.imageTransforms is not None:
			loadedFrame = self.imageTransforms(Image.open(frameDir + str(frameName) + FRAME_EXTN))
		else:
			loadedFrame = transforms.ToTensor(Image.open(frameDir + str(frameName) + FRAME_EXTN))
		actionLabel = int(actionLabel)
		return loadedFrame, actionLabel # the chosen frames from the video and the action label



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

	def __init__(self, nActionClasses, nEpochs, lr, momentumVal, trainLoader, testLoader, lrMilestones, ckpLoc, gpu = False):
		"""
		Initialize the model
		"""
		super(SpatialNetwork, self).__init__()
		self.nActionClasses = nActionClasses
		self.nEpochs = nEpochs
		self.lr = lr
		self.trainLoader = trainLoader
		self.testLoader = testLoader
		self.lrMilestones = lrMilestones
		self.gpu = gpu
		if self.gpu:
			print "GPU available!"
		else:
			print "No GPU available!"
		# get a VGG16 model pretrained with Imagenet; load it onto the graphic memory
		self.model = models.vgg16(pretrained = True)
		self.model.features.require_grad = False # fix the feature weights
		# swap out the final layer; the magic numbers are VGG parameters
		self.model.classifier[6] = nn.Linear(in_features=4096, out_features = nActionClasses, bias = True)
		self.criterion = nn.CrossEntropyLoss().cuda() if self.gpu else nn.CrossEntropyLoss() # set the loss function
		# a simple SGD optimizer
		self.optimizer = tch.optim.SGD(self.model.parameters(), self.lr, momentum = momentumVal) 
		self.startEpoch = 0 # initialize to the first epoch
		# a multi-step scheduler that changes the learning rate as per a fixed schedule
		self.scheduler = slr.MultiStepLR(self.optimizer, lrMilestones, gamma = 0.1, last_epoch = -1)
		self.highestPrecision = 0.0
		self.isBest = False
		if not ckpLoc.endswith("/"):
			ckpLoc += "/"
		checkAndMakeDirectories(ckpLoc) # create the checkpoint folder if it doesn't exist already
		self.ckpLoc = ckpLoc
		self.resumeLoc = self.ckpLoc + SPATIAL_CKP_FILE
		self.model = self.model.cuda() if self.gpu else self.model


	def train(self):
		"""
		Train for an epoch.
		"""
		self.model.train() # switch to train mode
		startTime = time.time()
		for iBatch, (data, label) in enumerate(self.trainLoader):
			if self.gpu:
				labelVar = ag.Variable(label.cuda(async = True))
				ip = ag.Variable(data.cuda(async = True))
			else:
				labelVar = ag.Variable(label)
				ip = ag.Variable(data)
			op = self.model(ip)
			loss = self.criterion(op, labelVar)
			self.optimizer.zero_grad()
			loss.backward()
			self.optimizer.step()

		torch.nn.utils.clip_grad_norm_(self.model.classifier.parameters(), max_norm=1.0)
		endTime = time.time()
		duration = endTime - startTime
		print "Epoch %d completed in %lf seconds" % (self.epoch, duration)
		self.save()


	def validate(self):
		"""
		Validate the model.
		"""
		self.model.eval() # switch to evaluation mode
		correct = 0
		total = len(self.testLoader.dataset)
		loss = 0
		with tch.no_grad():
			for iBatch, (data, label) in enumerate(self.testLoader):
				if self.gpu:
					labelVar = ag.Variable(label.cuda(async = True))
					ip = ag.Variable(data.cuda(async = True))
				else:
					labelVar = ag.Variable(label)
					ip = ag.Variable(data)
				op = self.model(ip)
				# print self.model.classifier[0].weight
				# print op.shape
				loss += self.criterion(op, labelVar) # total loss
				pred = op.max(1, keepdim=True)[1] # get the index of the max log-probability
				# print op.shape
				# print labelVar.shape
				correct += pred.eq(labelVar.view_as(pred)).sum().item()
		print self.epoch, total, correct, loss.item()
		print "Validation for epoch %d: total = %d, correct = %d, loss = %lf" % (self.epoch, total, correct, loss.item())
		return (correct / total), loss


	def resume(self):
		"""
		Resume training from a checkpoint, if found.
		"""
		if not (self.resumeLoc and os.path.isfile(self.resumeLoc)):
			print "No checkpoints found; starting from scratch!"
			return False
		print "Resuming training from checkpoint file: %s" % (self.resumeLoc)
		checkpoint = tch.load(self.resumeLoc)
		self.startEpoch = checkpoint["epoch"] + 1
		self.highestPrecision = checkpoint["highestPrecision"]
		self.model.load_state_dict(checkpoint["model"])
		self.optimizer.load_state_dict(checkpoint["optimizer"])
		self.scheduler = slr.MultiStepLR(self.optimizer, self.lrMilestones, gamma = 0.1, last_epoch = self.startEpoch)
		print "Loaded checkpoint: starting from epoch: %d" % (self.startEpoch)
		return True


	def save(self):
		"""
		Save the model state for future reference.
		"""
		makeCheckpoint({
			"epoch" : self.epoch,
			"model" : self.model.state_dict(),
			"highestPrecision" : self.highestPrecision,
			"optimizer" : self.optimizer.state_dict()
			}, self.isBest, self.ckpLoc + SPATIAL_CKP_FILE, self.ckpLoc + SPATIAL_BEST_FILE)
		return


	def execute(self):
		"""
		Execute all epochs. Each epoch consists of training 
		on all data followed by a validation step.
		"""
		found = self.resume()
		for self.epoch in range(self.startEpoch, self.nEpochs):
			# if not found:
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
	trainDataset = SpatialDataset(VIDEOLIST_TRAIN, FRAMES_DIR_TRAIN, imageTransforms, frameSampleSize = 2, actionLabelLoc = ACTIONLABEL_FILE)
	trainDataLoader = getDataLoader(trainDataset, batchSize = 5)
	# the same image transforms for the test data as well
	testDataset = SpatialDataset(VIDEOLIST_TEST, FRAMES_DIR_TEST, imageTransforms, mode = "test", actionLabelLoc = ACTIONLABEL_FILE)
	testDataLoader = getDataLoader(testDataset, batchSize = 2)
	gpu = tch.cuda.is_available()
	net = SpatialNetwork(NACTION_CLASSES, NEPOCHS, INITIAL_LR, MOMENTUM_VAL, trainDataLoader, testDataLoader, MILESTONES_LR, CHECKPOINT_DIR, gpu = False)
	net.execute()


if __name__ == '__main__':
	main()