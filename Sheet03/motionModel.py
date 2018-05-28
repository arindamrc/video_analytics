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
import itertools
from util import *

# Some parameters
VIDEO_FRAME_SAMPLE_RATE = 10
VIDEO_INPUT_FLOW_COUNT = 10
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
X_PREFIX_FLOW = "flow_x_"
Y_PREFIX_FLOW = "flow_y_"


class MotionDataset(Dataset):
	"""
	Inherits from the torch Dataset. Provides 2*L flow-frames 
	starting from a random frame from each video upon each invocation.
	Here flowSampleSize = L.
	"""

	def __init__(self, videoListLoc, rootDir, imageTransforms = None, flowSampleSize = VIDEO_INPUT_FLOW_COUNT, mode = "train", actionLabelLoc = None):
		"""
		The parameters mean:
		videoListLoc : The file with the video list.
		rootDir : The directory which has all saved flow frames per actionCategory per video.
		transforms : A list of torch image transforms to apply to each flow frame.
		flowSampleSize : 0.5 * No. of flow frames to get from each video (unifromly).
		(The starting frame is randomly selected)
		"""
		super(MotionDataset, self).__init__()
		if not rootDir.endswith("/"):
			self.rootDir = rootDir + "/"
		else:
			self.rootDir = rootDir
		self.imageTransforms = imageTransforms
		self.flowSampleSize = flowSampleSize
		self.mode = mode
		with open(videoListLoc, "r") as videoListFile:
			self.videoList = [line for line in videoListFile]
		if actionLabelLoc is None:
			raise ValueError("Action label dictionary required!")
		with open(actionLabelLoc, "r") as actionLabelFile:
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
		flowDir = self.rootDir + actionCategory + "/" + videoName + "/"
		# get the number of frames in the folder
		nFlows = len([fName for fName in os.listdir(flowDir)]) / 2 # there x and y flows
		iFlowFrame = random.randint(0, nFlows - self.flowSampleSize - 1) # load a random flow frame
		xFlowFrames = [self.rootDir + videoName + "/" + X_PREFIX_FLOW + str(idx).zfill(4) + FRAME_EXTN for idx in range(iFlowFrame, iFlowFrame + self.flowSampleSize)]
		xFlowFrames = [self.rootDir + videoName + "/" + Y_PREFIX_FLOW + str(idx).zfill(4) + FRAME_EXTN for idx in range(iFlowFrame, iFlowFrame + self.flowSampleSize)]
		# combine the two lists alternatingly
		flowFrames = list(it.next() for it in itertools.cycle([iter(xFlowFrames), iter(yFlowFrames)]))
		# load the flow frames
		if self.imageTransforms is not None:
			loadedFrames = [self.imageTransforms(Image.open(frame)) for frame in flowFrames]
		else:
			loadedFrames = [transforms.ToTensor(Image.open(frame)) for frame in flowFrames]
		loadedFrames = tch.stack(loadedFrames) # convert the list of flow frames into a tensor
		actionLabel = int(actionLabel)
		return loadedFrame, actionLabel # the chosen flow frames from the video and the action label



class MotionNetwork(object):
	"""
	A wrapper for the motion stream.
	"""

	def __init__(self, nActionClasses, flowSampleSize, nEpochs, lr, momentumVal, trainLoader, testLoader, lrMilestones, ckpLoc, gpu = False):
		"""
		Initialize the model
		"""
		super(MotionNetwork, self).__init__()
		self.nActionClasses = nActionClasses
		self.nEpochs = nEpochs
		self.lr = lr
		self.trainLoader = trainLoader
		self.testLoader = testLoader
		self.lrMilestones = lrMilestones
		self.flowSampleSize = flowSampleSize
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


	def __modifyFirstLayer(self, vggNet):
		layerOne = vggNet.features[0]
		avg = 0
		for inChannel in range(layerOne.in_channels):
			avg += layerOne.weight[:,inChannel,:,:]
		avg /= layerOne.in_channels
		newLayerOne = nn.Conv2d(self.flowSampleSize * 2, layerOne.out_channels, kernel_size = layerOne.kernel_size, padding = layerOne.padding)



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
	imageTransforms = getTransforms()
	trainDataset = MotionDataset(VIDEOLIST_TRAIN, FRAMES_DIR_TRAIN, imageTransforms, flowSampleSize = 2, actionLabelLoc = ACTIONLABEL_FILE)
	trainDataLoader = getDataLoader(trainDataset, batchSize = 5)
	# the same image transforms for the test data as well
	testDataset = MotionDataset(VIDEOLIST_TEST, FRAMES_DIR_TEST, imageTransforms, mode = "test", actionLabelLoc = ACTIONLABEL_FILE)
	testDataLoader = getDataLoader(testDataset, batchSize = 2)
	gpu = tch.cuda.is_available()
	net = MotionNetwork(NACTION_CLASSES, NEPOCHS, INITIAL_LR, MOMENTUM_VAL, trainDataLoader, testDataLoader, MILESTONES_LR, CHECKPOINT_DIR, gpu = False)
	net.execute()


if __name__ == '__main__':
	main()