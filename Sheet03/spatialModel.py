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
from utils import *
from parameters import * 


class SpatialDataset(Dataset):
	"""
	Inherits from the torch Dataset. Provides a random frame
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
			self.videoList = [line for line in videoListFile][:20]
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
		frameDir = self.rootDir + actionCategory + "/" + videoName + "/"
		# get the number of frames in the folder
		nFrames = len([frameName for frameName in os.listdir(frameDir)])
		frameName = random.randint(0, nFrames - 1) # load a random frame
		if self.imageTransforms is not None:
			loadedFrame = self.imageTransforms(Image.open(frameDir + str(frameName) + FRAME_EXTN))
		else:
			loadedFrame = transforms.ToTensor(Image.open(frameDir + str(frameName) + FRAME_EXTN))
		actionLabel = int(actionLabel)
		return loadedFrame, actionLabel, videoName # the chosen frames from the video and the action label



class SpatialNetwork(object):
	"""
	A wrapper for the spatial stream.
	"""

	def __init__(self, nActionClasses, nEpochs, lr, momentumVal, descriptorDim, trainLoader, testLoader, lrMilestones, ckpLoc, gpu = False):
		"""
		Initialize the model
		"""
		super(SpatialNetwork, self).__init__()
		self.nActionClasses = nActionClasses
		self.nEpochs = nEpochs
		self.lr = lr
		self.trainLoader = trainLoader
		self.totalTrain = len(self.trainLoader.dataset)
		self.testLoader = testLoader
		self.totalTest = len(self.testLoader.dataset)
		self.lrMilestones = lrMilestones
		self.descriptorDim = descriptorDim
		self.gpu = gpu
		if self.gpu:
			print "GPU available!"
		else:
			print "No GPU available!"
		# get a VGG16 model pretrained with Imagenet; load it onto the graphic memory
		self.model = models.vgg16(pretrained = True)
		self.model.features.require_grad = False # fix the feature weights
		# swap out the final layer
		self.__swapClassifier__()
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
		self.features = self.model.features
		self.classifierList = list(self.model.classifier)
		self.classifierLen = len(self.classifierList)
		# video level descriptor holders
		self.trainDict = {} 
		self.testDict = {}
		self.model = nn.DataParallel(self.model) if self.gpu else self.model


	def __swapClassifier__(self):
		"""
		Swap the classifier layer of vgg16 for a custom one. 
		The video descriptor's dimension can be set.
		"""
		self.model.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, self.descriptorDim),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(self.descriptorDim, self.nActionClasses),
        )




	def train(self):
		"""
		Train for an epoch.
		"""
		self.model.train() # switch to train mode
		startTime = time.time()
		featureVectors = None
		for iBatch, (data, labels, videoNames) in enumerate(self.trainLoader):
			if self.gpu:
				labelVar = ag.Variable(labels.cuda(async = True))
				ip = ag.Variable(data.cuda(async = True))
			else:
				labelVar = ag.Variable(labels)
				ip = ag.Variable(data)
			op = self.features(ip)
			op = op.view(op.size(0), -1)
			for cl in self.classifierList[:(self.classifierLen - 1)]: # evaluate till second last layer
				op = cl(op)
			featureVectors = op # keep the second last layer's output as the feature vector
			for cl in self.classifierList[(self.classifierLen - 1):]: # continue till last layer
				op = cl(op)
			loss = self.criterion(op, labelVar)
			self.optimizer.zero_grad()
			loss.backward()
			self.optimizer.step()
			# collate video level features
			for i in range(len(featureVectors)):
				if videoNames[i] in self.trainDict:
					self.trainDict[videoNames[i]][0].update(featureVectors[i])
				else:
					self.trainDict[videoNames[i]] = (AverageMeter(), labels[i])
					self.trainDict[videoNames[i]][0].update(featureVectors[i])

		# torch.nn.utils.clip_grad_norm_(self.model.classifier.parameters(), max_norm=1.0)
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
		loss = 0
		with tch.no_grad():
			for iBatch, (data, labels, videoNames) in enumerate(self.testLoader):
				if self.gpu:
					labelVar = ag.Variable(labels.cuda(async = True))
					ip = ag.Variable(data.cuda(async = True))
				else:
					labelVar = ag.Variable(labels)
					ip = ag.Variable(data)
				op = self.features(ip)
				op = op.view(op.size(0), -1)
				for cl in self.classifierList[:(self.classifierLen - 1)]: # evaluate till second last layer
					op = cl(op)
				featureVectors = op # keep the second last layer's output as the feature vector
				for cl in self.classifierList[(self.classifierLen - 1):]: # continue till last layer
					op = cl(op)
				loss += self.criterion(op, labelVar) # total loss
				pred = op.max(1, keepdim=True)[1] # get the index of the max log-probability
				correct += pred.eq(labelVar.view_as(pred)).sum().item()
				# collate video level features
				for i in range(len(featureVectors)):
					if videoNames[i] in self.testDict:
						self.testDict[videoNames[i]][0].update(featureVectors[i])
					else:
						self.testDict[videoNames[i]] = (AverageMeter(), labels[i])
						self.testDict[videoNames[i]][0].update(featureVectors[i])

		print "Validation for epoch %d: total = %d, correct = %d, loss = %lf" % (self.epoch, self.totalTest, correct, loss.item())
		return (correct / self.totalTest), loss


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
			savePerformance(precision, loss.data.item(), SPATIAL_PERFORMANCE_LOC) # save epoch performance
			# save video level descriptors
			saveVideoDescriptors(self.trainDict, SPATIAL_TRAIN_CSV_LOC)
			saveVideoDescriptors(self.testDict, SPATIAL_TEST_CSV_LOC)


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
	net = SpatialNetwork(NACTION_CLASSES, NEPOCHS, INITIAL_LR, MOMENTUM_VAL, VIDEO_DESCRIPTOR_DIM, trainDataLoader, testDataLoader, MILESTONES_LR, CHECKPOINT_DIR, gpu = False)
	net.execute()


if __name__ == '__main__':
	main()