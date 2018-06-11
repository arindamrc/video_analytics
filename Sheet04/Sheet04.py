from __future__ import division
import numpy as np
import os
import re
import scipy.stats as stats
from collections import deque
import sys

GMM_DIM = 64
EPSILON = 0.000001


HMM_DEF_DICT = "/home/arc/VA_Assignments/Sheet04/exc1/hmm_definition.dict"
HMM_DEF_VECT = "/home/arc/VA_Assignments/Sheet04/exc1/hmm_definition.vector"
HMM_ST_DEF_VECT = "/home/arc/VA_Assignments/Sheet04/exc1/hmm_state_definition.vector"
ST_INIT_SUFFIX = "initStates.npy"
SEQUENCE_LOC = "/home/arc/VA_Assignments/Sheet04/exc2/train_samples/"


# Some utilities

def sortAlphaNumerically(someList):
	"""
	Sort a string list. ["AL13", "AL3", "AA14", "AA4"] would become ["AA4", "AA14", "AL3", "AL13"]
	"""
	convert = lambda text: int(text) if text.isdigit() else text
	alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
	return sorted(someList, key = alphanum_key)

def fixZeros(arr):
	"""
	A fix to prevent division by zero.
	"""
	arr = arr.copy()
	arr[abs(arr) < EPSILON] = 1.0
	return arr

def sumNormalize(arr):
	"""
	Normalize the array based on sum.
	"""
	s = np.sum(arr)
	if s == 0.0:
		s = 1.0
	return arr / s

def normalizePerRow(arr2d, checkZeros = True):
	"""
	Ensures that each row sums to 1.0.
	"""
	rowSums = arr2d.sum(axis=1)
	if checkZeros:
		rowSums = fixZeros(rowSums) # prevent division by zero
	normalized = arr2d / rowSums[:, np.newaxis]
	return normalized

###########################################################################################


def getActivities(actLoc = HMM_DEF_DICT):
	"""
	Get a list of activities.
	"""
	activities = []
	with open(actLoc,"r") as actFile:
		for line in actFile:
			activities.append(line.strip())
	return activities


def getActivityStates(activities, actDefLoc = HMM_DEF_VECT, subActLoc = HMM_ST_DEF_VECT):
	"""
	Get a list of sub-activities indexed by activity.
	"""
	stateNames = []
	stateIndices = []
	stateIndex = 0
	with open(actDefLoc, "r") as actDefFile, open(subActLoc, "r") as subActDefFile:
		for activity in range(len(activities)):
			stateCount = int(actDefFile.readline().strip())
			stateNameList = []
			stateIndexList = []
			for state in range(stateCount):
				stateNameList.append(subActDefFile.readline().strip())
				stateIndexList.append(stateIndex)
				stateIndex += 1
			stateNames.append(stateNameList)
			stateIndices.append(stateIndexList)
	return stateNames, stateIndices


def getSequenceFiles(activity, sequenceLoc = SEQUENCE_LOC):
	"""
	Get sequence file names for the gitven activity.
	"""
	assert os.path.isdir(sequenceLoc)
	seqFeatures = [os.path.join(sequenceLoc, fname) for fname in os.listdir(sequenceLoc) if os.path.isfile(os.path.join(sequenceLoc, fname)) and fname.startswith(activity) and not fname.endswith(ST_INIT_SUFFIX)]
	seqInits = [os.path.join(sequenceLoc, fname) for fname in os.listdir(sequenceLoc) if os.path.isfile(os.path.join(sequenceLoc, fname)) and fname.startswith(activity) and fname.endswith(ST_INIT_SUFFIX)]
	assert len(seqFeatures) == len(seqInits)
	return sortAlphaNumerically(seqFeatures), sortAlphaNumerically(seqInits)


def getTransitionProbs(initStates, stateIndices):
	"""
	Constructs the initial state transition probability matrix based on the supplied initial states.
	"""
	states, stateCounts = np.unique(initStates, return_counts = True) # state indices
	nStates = len(stateIndices)
	pTransitions = np.zeros((nStates, nStates), dtype = float) # initialize the N x N state transition matrix.
	# first, the self transitions
	for i in range(nStates):
		state = stateIndices[i]
		if state in states:
			pTransitions[i,i] = (stateCounts[i] - 1) / len(initStates) 
		else:
			pTransitions[i,i] = 0.0
	# Now the transition to other states; Assumption: states always appear in order
	for i in range(nStates):
		state = stateIndices[i]
		if state in states:
			for j in range(i + 1, len(stateCounts)):
				pTransitions[i, j] = stateCounts[j] / len(initStates)
		else:
			for j in range(i + 1, nStates):
				pTransitions[i, j] = 0.0
	return pTransitions



def getObservationProbs(initStates, stateIndices):
	"""
	Constructs the intial observation probability matrix based on supplied initial states and the sequence of observations.
	"""
	# states, stateCounts = np.unique(initStates, return_counts = True) # state indices
	# # compensate for missing states
	# mismatches = np.where(states != stateIndices)
	# for i in mismatches:
	# 	np.insert(stateCounts, i, 0)
	# pObservations = np.zeros((len(stateIndices), len(initStates)), dtype = float) # initialize the N x T matrix
	# stateIndexDict = {stateIndices[i] : i for i in range(len(stateIndices))}
	# for i in range(len(initStates)):
	# 	st = initStates[i]
	# 	pObservations[stateIndexDict[st], i] = 1.0
	pObservations = np.ones((len(stateIndices), len(initStates)), dtype = float) # initialize the N x T matrix
	pObservations = normalizePerRow(pObservations) # normalize observation probabilities per state
	return pObservations


def getStateProbs(initStates, stateIndices):
	"""
	Constructs the initial state probability matrix. This is just the state distribution.
	"""
	states, stateCounts = np.unique(initStates, return_counts = True) # state indices
	pStates = np.zeros((len(stateIndices), 1), dtype = float)
	for i in range(len(stateIndices)):
		state = stateIndices[i]
		if state in states:
			pStates[i, 0] = stateCounts[i] / len(initStates)
	return pStates



class BaumWelch(object):
	"""
	The Baum-Welch algorithm. Parameters are:

	observations: The sequence of observations or feature vectors O(1...T)
	pTransitions: An N x N matrix giving the state transition probabilities. N -> no. of states.
	pObservations: An N x T- matrix giving the observation probabilities of each of the T features at each state.
	pStates: The probability of each state occurring. N x 1.
	"""
	def __init__(self, activity, observations, pTransitions, pObservations, pStates):
		super(BaumWelch, self).__init__()
		self.activity = activity
		self.observations = observations
		self.pTransitions = pTransitions
		self.pObservations = pObservations
		self.pStates = pStates
		self.stateCount = len(pStates)
		self.obsCount = observations.shape[0] # T
		self.dim = observations.shape[1]
		self.reset()


	def reset(self):
		# initialize forward and backward passes
		self.alpha = deque()
		self.beta = deque()
		# initialize the temporary variables
		self.gamma = []
		self.eta = []
		self.__temporaries__()
		

	def __passes__(self):
		# initialize the forward and backward passes
		alpha_t = self.pStates * self.pObservations[:, 0, np.newaxis]
		beta_t = np.ones((self.stateCount, 1), dtype = float)
		self.alpha.append(alpha_t)
		self.beta.append(beta_t)
		for t in range(self.obsCount - 1):
			for i in range(self.stateCount):
				sumAlpha = 0
				sumBeta = 0
				for j in range(self.stateCount):
					sumAlpha += alpha_t[j] * self.pTransitions[j,i]
					sumBeta += beta_t[j] * self.pTransitions[i,j] * self.pObservations[j,(self.obsCount - 1 - t)]
				alpha_t[i] = self.pObservations[i, t+1] * sumAlpha
				beta_t[i] = sumBeta
			alpha_t = sumNormalize(alpha_t)
			beta_t = sumNormalize(beta_t)
			self.alpha.append(alpha_t)
			self.beta.appendleft(beta_t)
		return


	def __temporaries__(self):
		"""
		Calculate the temporaries gamma and eta: results of baum-welch.
		"""
		self.__passes__() # make the forward and backward passes.
		gamma_sum = np.zeros_like(self.alpha[0])
		for t in range(self.obsCount):
			gamma_t = np.zeros((self.stateCount, 1))
			divisor = np.sum(self.alpha[t] * self.beta[t], axis = 0)[0] # single element array
			if abs(divisor) < EPSILON:
				divisor = 1.0
			for i in range(self.stateCount):
				gamma_t[i] = (self.alpha[t][i] * self.beta[t][i]) / divisor
			self.gamma.append(gamma_t)
		for t in range(self.obsCount - 1):
			beta_t = fixZeros(self.beta[t])
			eta_t = np.zeros((self.stateCount, self.stateCount), dtype = float)
			for i in range(self.stateCount):
				for j in range(self.stateCount):
					eta_t[i,j] = self.gamma[t][i] * self.pTransitions[i,j] * self.pObservations[j, t+1] * self.beta[t+1][j] / beta_t[i]
			self.eta.append(eta_t)
		return


	def getTemporaries(self):
		"""
		Return the calculated temporaries.
		"""
		return self.gamma, self.eta



class LearnGaussians(object):
	"""Calculates the parameters of individual Gaussians using the Baum-Welch algorithm."""
	
	def __init__(self, subActivity, stateIndices, seqLoc):
		"""
		Initialize with the subActivity name and the location of all it's sequences.
		"""
		super(LearnGaussians, self).__init__()
		self.subActivity = subActivity
		self.seqFeatures, self.initFiles = getSequenceFiles(subActivity, seqLoc)
		self.stateIndices = stateIndices
		self.stateCount = len(stateIndices)
		self.bwList = [] # for each sequence
		self.__intitBW__()


	def __intitBW__(self):
		"""
		Compute the initial probabilities for a sequence based on the *_initStates.py files.
		Initialize Baum-Welch for each sequence in the list.
		"""
		i = 0
		observations = None
		self.sequences = []
		for obsFile, initFile in zip(self.seqFeatures, self.initFiles):
			observations = np.load(obsFile)
			self.sequences.append(observations)
			initStates = np.ravel(np.load(initFile))
			pTransitions = getTransitionProbs(initStates, self.stateIndices)
			pObservations = getObservationProbs(initStates, self.stateIndices)
			self.pStates = getStateProbs(initStates, self.stateIndices)
			bw = BaumWelch(self.subActivity + str(i), observations, pTransitions, pObservations, self.pStates)
			self.bwList.append(bw)
			i += 1
		self.dim = self.sequences[0].shape[1]


	def __mu__(self):
		"""
		Get the list of means of all Gaussians.
		"""
		mus = []
		for i in range(self.stateCount):
			mu = np.zeros(self.dim, dtype = float)
			sumGamma = 0
			for nSeq in range(len(self.seqFeatures)):
				observations = self.sequences[nSeq] # a sequence of T observations
				gamma, _ = self.bwList[nSeq].getTemporaries()
				for t in range(len(gamma)):
					observation_t = observations[t].reshape(self.dim)
					mu += gamma[t][i] * observation_t
					sumGamma += gamma[t][i]
			sumGamma = 1.0 if abs(sumGamma) < EPSILON else sumGamma
			mu = mu / sumGamma
			mus.append(mu)
		return mus


	def __sigma__(self, mus):
		"""
		Get the list standard deviation of the gaussians.
		"""
		sigmas = []
		for i in range(self.stateCount):
			sigma = np.zeros((self.dim, self.dim), dtype = float)
			sumGamma = 0
			for nSeq in range(len(self.seqFeatures)):
				observations = self.sequences[nSeq] # a sequence of T observations
				gamma, _ = self.bwList[nSeq].getTemporaries()
				for t in range(len(gamma)):
					observation_t = observations[t].reshape(self.dim)
					diff = (observation_t - mus[i]).reshape(self.dim, 1)
					sigma += gamma[t][i] * diff * diff.T
					sumGamma += gamma[t][i]
			sumGamma = 1.0 if abs(sumGamma) < EPSILON else sumGamma
			sigma = sigma / sumGamma
			sigmas.append(sigma)
		return sigmas


	def __updateTransitionProbs__(self):
		"""
		Update the transition probabilities for the baum-welch algorithm.
		"""
		pTransitions = np.zeros((self.stateCount, self.stateCount), dtype = float)
		for i in range(self.stateCount):
			sumGamma = 0
			for j in range(self.stateCount):
				for nSeq in range(len(self.seqFeatures)):
					gamma, eta = self.bwList[nSeq].getTemporaries()
					for t in range(len(eta)):
						pTransitions[i,j] += eta[t][i,j]
						sumGamma += gamma[t][i]
		print pTransitions
		pTransitions = normalizePerRow(pTransitions, checkZeros = False)
		print pTransitions
		for bw in self.bwList:
			bw.pTransitions = pTransitions
		return


	def __updateObservationProbs__(self, mus, sigmas):
		"""
		Update the observation probabilities for the baum-welch algorithm.
		"""
		for nSeq in range(len(self.seqFeatures)):
			observations = self.sequences[nSeq] # a sequence of T observations
			pObservations = np.zeros((self.stateCount, len(observations)))
			for j in range(self.stateCount):
				mu = mus[j]
				sigma = sigmas[j]
				singular = False
				if abs(np.linalg.det(sigma)) == 0:
					singular = True
				for t in range(len(observations)):
					observation_t = observations[t]
					if singular:
						pObservations[j, t] = 0.0
					else:
						pObservations[j, t] = stats.multivariate_normal.pdf(observation_t, mean = mu, cov = np.square(sigma))
			pObservations = normalizePerRow(pObservations, checkZeros = False)
			self.bwList[nSeq].pObservations = pObservations
		return


	def __resetBaumWelch__(self):
		"""
		Reset Baum-Welch for next iteration
		"""
		for bw in self.bwList:
			bw.reset()
		return



	def run(self):
		"""
		Execute the iterative learning procedure.
		"""
		mus = self.__mu__()
		sigmas = self.__sigma__(mus)
		self.__updateTransitionProbs__()
		self.__updateObservationProbs__(mus, sigmas)
		self.__resetBaumWelch__()
		return mus, sigmas




		
		

def main():
	activities = getActivities(HMM_DEF_DICT)
	stateNames, stateIndices = getActivityStates(activities)
	lg = LearnGaussians(activities[0], stateIndices[0], SEQUENCE_LOC)
	lg.run()
	


if __name__ == '__main__':
	main()