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


def normalizePerRow(arr2d):
	"""
	Ensures that each row sums to 1.0.
	"""
	rowSums = arr2d.sum(axis=1)
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


def getTransitionProbs2(initStates, stateIndices):
	"""
	Constructs the initial state transition probability matrix based on the supplied initial states.
	"""
	states, stateCounts = np.unique(initStates, return_counts = True) # state indices
	# compensate for missing states
	print "states before"
	print states
	states = np.pad(states, pad_width = (0, len(stateIndices) - len(states)), mode = "constant", constant_values = -1)
	print "states after"
	print states
	print "stateIndices"
	print stateIndices
	print "states counts before"
	print stateCounts
	mismatches = np.where(states != stateIndices)
	print "mismatches"
	print mismatches[0]
	print "len(mismatches[0])"
	print len(mismatches[0])
	if len(mismatches) > 0:
		stateCounts = np.insert(stateCounts, mismatches[0], 0)
	print "states counts after"
	print stateCounts
	stateIndexDict = {stateIndices[i] : i for i in range(len(stateIndices))}
	nStates = len(stateIndices) # state count = gaussian component count
	pTransitions = np.zeros((nStates, nStates), dtype = float) # initialize the N x N state transition matrix.
	# first, the self transitions
	for i in range(nStates):
		if stateCounts[i] == 0:
			pTransitions[i,i] = 0.0
		else:
			pTransitions[i,i] = (stateCounts[i] - 1) / len(initStates) 
	for i in range(len(initStates)):
		nxtStateCounts = {st:0 for st in stateIndices if st != initStates[i]}
		for j in range(i, len(initStates)): # start from the current state because HMM is left to right ordered
			if initStates[j] == initStates[i]:
				continue # skip for self transitions
			nxtStateCounts[initStates[j]] += 1
		for st, count in nxtStateCounts.iteritems():
			pTransitions[i, stateIndexDict[st]] = count / len(initStates)
	pTransitions = normalizePerRow(pTransitions) # transition probabilities from a state should sum to 1.0
	return pTransitions


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
	states, stateCounts = np.unique(initStates, return_counts = True) # state indices
	# compensate for missing states
	mismatches = np.where(states != stateIndices)
	for i in mismatches:
		np.insert(stateCounts, i, 0)
	pObservations = np.zeros((len(stateIndices), len(initStates)), dtype = float) # initialize the N x T matrix
	stateIndexDict = {stateIndices[i] : i for i in range(len(stateIndices))}
	for i in range(len(initStates)):
		st = initStates[i]
		pObservations[stateIndexDict[st], i] = 1.0
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
			alpha_t = self.pObservations[:, t+1, np.newaxis] * np.sum(alpha_t * self.pTransitions, axis = 0, keepdims = True).T # forward pass
			beta_t = np.sum(beta_t * self.pTransitions.T * self.pObservations[:, (self.obsCount - 1 - t), np.newaxis], axis = 0, keepdims = True).T # backward pass
			self.alpha.append(alpha_t)
			self.beta.appendleft(beta_t)


	def __temporaries__(self):
		"""
		Calculate the temporaries gamma and eta: results of baum-welch.
		"""
		self.__passes__() # make the forward and backward passes.
		gamma_sum = np.zeros_like(self.alpha[0])
		# print "self.obsCount"
		# print self.obsCount
		for t in range(self.obsCount):
			gamma_t = self.alpha[t] * self.beta[t]
			self.gamma.append(gamma_t)
			gamma_sum += gamma_t
		for t in range(self.obsCount - 1):
			beta_t = fixZeros(self.beta[t])
			eta_t = np.zeros((self.stateCount, self.stateCount), dtype = float)
			for i in range(self.stateCount):
				for j in range(self.stateCount):
					eta_t[i,j] = self.gamma[t][i] * self.pTransitions[i,j] * self.pObservations[j, t+1] * self.beta[t+1][j] / beta_t[i]
			self.eta.append(eta_t)
		for t in range(self.obsCount):
			gamma_sum = fixZeros(gamma_sum)
			self.gamma[t] /= gamma_sum
		return self.gamma, self.eta


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
		for obsFile, initFile in zip(self.seqFeatures, self.initFiles):
			observations = np.load(obsFile)
			initStates = np.ravel(np.load(initFile))
			# print initFile
			# print initStates
			pTransitions = getTransitionProbs(initStates, self.stateIndices)
			pObservations = getObservationProbs(initStates, self.stateIndices)
			self.pStates = getStateProbs(initStates, self.stateIndices)
			bw = BaumWelch(self.subActivity + str(i), observations, pTransitions, pObservations, self.pStates)
			self.bwList.append(bw)
			i += 1
		self.dim = observations.shape[1]


	def __mu__(self):
		"""
		Get the list of means of all Gaussians.
		"""
		sumGamma = None
		sumNumerator = None
		for nSeq in range(len(self.seqFeatures)):
			observations = np.load(self.seqFeatures[nSeq]) # a sequence of T observations
			bw = self.bwList[nSeq] # baum-welch
			gamma, eta = bw.getTemporaries() # each a sequence of T elements
			for t in range(len(eta)):
				sumGamma = gamma[t] if sumGamma is None else sumGamma + gamma[t]
				sumNumerator = gamma[t] * observations[t] if sumNumerator is None else sumNumerator + gamma[t] * observations[t]
		sumGamma = fixZeros(sumGamma)
		return sumNumerator / sumGamma


	def __sigma__(self, mu):
		"""
		Get the list standard deviation of the gaussians.
		"""
		sumGamma = np.zeros((self.stateCount, 1), dtype = float)
		sigmas = np.zeros((self.dim, self.dim, self.stateCount), dtype = float)
		for nSeq in range(len(self.seqFeatures)):
			observations = np.load(self.seqFeatures[nSeq]) # a sequence of T observations
			bw = self.bwList[nSeq] # baum-welch
			gamma, eta = bw.getTemporaries() # each a sequence of T elements
			# print "len(gamma)"
			# print len(gamma)
			# print "len(eta)"
			# print len(eta)
			# print "len(observations)"
			# print len(observations)
			for t in range(len(eta)):
				observation = observations[t, np.newaxis].T
				# print "observation.shape"
				# print observation.shape
				sumGamma += gamma[t]
				for i in range(self.stateCount):
					# print "mu[i].shape"
					# print mu[i].shape
					var = np.matmul((observation - mu[i]),(observation - mu[i]).T)
					# print "var.shape"
					# print var.shape
					sigmas[:,:,i] += gamma[t][i] * var
		for i in range(self.stateCount):
			divisor = fixZeros(sumGamma[i])
			sigmas[:,:,i] = sigmas[:,:,i] / divisor
		return sigmas


	def __updateTransitionProbs__(self):
		"""
		Update the transition probabilities for the baum-welch algorithm.
		"""
		sumGamma = None
		sumEta = None
		for nSeq in range(len(self.seqFeatures)):
			observations = np.load(self.seqFeatures[nSeq]) # a sequence of T observations
			bw = self.bwList[nSeq] # baum-welch
			gamma, eta = bw.getTemporaries() # each a sequence of T elements
			for t in range(len(eta)):
				sumGamma = gamma[t] if sumGamma is None else sumGamma + gamma[t]
				sumEta = eta[t] if sumEta is None else sumEta + eta[t]
		pTransitions = sumEta / sumGamma
		for bw in self.bwList:
			bw.pTransitions = pTransitions
		return


	def __updateObservationProbs__(self, mus, sigmas):
		"""
		Update the observation probabilities for the baum-welch algorithm.
		"""
		for nSeq in range(len(self.seqFeatures)):
			observations = np.load(self.seqFeatures[nSeq]) # a sequence of T observations
			pObservations = np.zeros((len(mus), len(observations)))
			for nGaussian in range(len(mus)):
				mu = mus[nGaussian]
				sigma = sigmas[:,:,nGaussian]
				if abs(np.linalg.det(sigma)) == 0:
					singular = True
				for t in range(len(observations)):
					observation = observations[t]
					if singular:
						pObservations[nGaussian, t] = 0.0
					else:
						pObservations[nGaussian, t] = stats.multivariate_normal.pdf(observation, mean = mu, cov = np.square(sigma))
			self.bwList[nSeq].pObservations = pObservations
			print pObservations
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