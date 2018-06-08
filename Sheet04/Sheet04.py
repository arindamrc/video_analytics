from __future__ import division
import numpy as np
import os
import re
import scipy.stats as stats
from collections import deque

GMM_DIM = 64


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


def normalizePerRow(arr2d):
	"""
	Ensures that each row sums to 1.0.
	"""
	rowSums = arr2d.sum(axis=1)
	return arr2d / rowSums[:, np.newaxis]

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


def getSubActivities(activities, actDefLoc = HMM_DEF_VECT, subActLoc = HMM_ST_DEF_VECT):
	"""
	Get a list of sub-activities indexed by activity.
	"""
	subActivities = []
	with open(actDefLoc, "r") as actDefFile, open(subActLoc, "r") as subActDefFile:
		for activity in range(len(activities)):
			stateCount = int(actDefFile.readline().strip())
			subActivityList = []
			for state in range(stateCount):
				subActivityList.append(subActDefFile.readline().strip())
			subActivities.append(subActivityList)
	return subActivities


def getSequenceFiles(activity, sequenceLoc = SEQUENCE_LOC):
	"""
	Get sequence file names for the gitven activity.
	"""
	assert os.path.isdir(sequenceLoc)
	seqFeatures = [fname for fname in os.listdir(sequenceLoc) if os.path.isfile(os.path.join(sequenceLoc, fname)) and fname.startswith(activity) and not fname.endswith(ST_INIT_SUFFIX)]
	seqInits = [fname for fname in os.listdir(sequenceLoc) if os.path.isfile(os.path.join(sequenceLoc, fname)) and fname.startswith(activity) and fname.endswith(ST_INIT_SUFFIX)]
	assert len(seqFeatures) == len(seqInits)
	return sortAlphaNumerically(seqFeatures), sortAlphaNumerically(seqInits)


def baumWelch(seqLoc, initLoc):
	features = np.load(seqLoc) # features representing each frame
	initStates = np.load(initLoc) # initial states per frame
	states = np.unique(initStates) # state indices
	nStates = len(states) # state count = gaussian component count
	pTrans = np.ones((nStates, nStates), dtype = np.float32) # state transition probabilities
	pTrans = pTrans / (nStates * nStates) # intialize to uniform transition probabilities


def getTransitionProbs(initStates):
	"""
	Constructs the initial state transition probability matrix based on the supplied initial states.
	"""
	states, stateCounts = np.unique(initStates, return_counts = True) # state indices
	stateIndices = {states[i] : i for i in range(len(states))}
	nStates = len(states) # state count = gaussian component count
	pTransitions = np.zeros((nStates, nStates), dtype = float) # initialize the N x N state transition matrix.
	for i in range(len(states)):
		# first, the self transitions
		pTransitions[i,i] = (stateCounts[i] - 1) / len(initStates) 
		nxtStateCounts = {st:0 for st in states if st != initStates[i]}
		for j in range(i, len(initStates)): # start from the current state because HMM is left to right ordered
			if initStates[j] == initStates[i]:
				continue # skip for self transitions
			nxtStateCounts[initStates[j]] += 1
		for st, count in nxtStateCounts.iteritems():
			pTransitions[i, stateIndices[st]] = count / len(initStates)
	pTransitions = normalizePerRow(pTransitions) # transition probabilities from a state should sum to 1.0
	return pTransitions


def getObservationProbs(initStates):
	"""
	Constructs the intial observation probability matrix based on supplied initial states and the sequence of observations.
	"""
	states, stateCounts = np.unique(initStates, return_counts = True) # state indices
	pObservations = np.zeros((nStates, len(initStates)), dtype = float) # initialize the N x T matrix
	stateIndices = {states[i] : i for i in range(len(states))}
	for i in range(len(initStates)):
		st = initStates[i]
		pObservations[stateIndices[st], i] = 1.0
	pObservations = normalizePerRow(pObservations) # normalize observation probabilities per state
	return pObservations


def getStateProbs(initStates):
	"""
	Constructs the initial state probability matrix. This is just the state distribution.
	"""
	states, stateCounts = np.unique(initStates, return_counts = True) # state indices
	pStates = np.zeros((len(states), 1), dtype = float)
	for i in range(len(states)):
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
	def __init__(self, observations, pTransitions, pObservations, pStates, muS, sigmaS):
		super(BaumWelch, self).__init__()
		self.observations = observations
		self.pTransitions = pTransitions
		self.pObservations = pObservations
		self.pStates = pStates
		self.stateCount = len(pStates)
		self.obsCount = observations.shape[0] # T
		self.dim = observations.shape[1]
		# self.muS = normalizePerRow(np.ones((self.stateCount, dims), type = float))
		# self.sigmaS = normalizePerRow(np.ones((self.stateCount, dims), type = float))
		# initialize forward and backward passes
		self.alpha = deque()
		self.beta = deque()
		# initialize the temporary variables
		self.gamma = []
		self.eta = []
		

	def __passes__(self):
		# initialize the forward and backward passes
		alpha_t = pStates * pObservations[:,0]
		beta_t = np.ones((stateCount, 1), dtype = float)
		self.alpha.append(alpha_t)
		self.beta.append(beta_t)
		for t in range(self.obsCount - 1):
			alpha_t = pObservations[:,t+1] * np.sum(alpha_t * pTransitions, axis = 0)
			beta_t = np.sum(beta_t * pTransitions * pObservations[:, (self.obsCount - t)], axis = 0)
			self.alpha.append(alpha_t)
			self.beta.appendleft(beta_t)


	def getTemporaries(self):
		"""
		Returns: The temporaries gamma and eta.
		"""
		gamma_sum = np.zeros_like(self.alpha[0])
		for t in range(self.obsCount):
			gamma_t = self.alpha[t] * self.beta[t]
			self.gamma.append(gamma_t)
			gamma_sum += gamma_t
		for t in range(self.obsCount - 1):
			eta_t = self.gamma[t] * self.pTransitions * pObservations[:,t+1] * self.beta[t+1]

		return self.gamma, self.eta
		pass
		

def main():
	activities = getActivities(HMM_DEF_DICT)
	print activities
	print getSubActivities(activities)
	print activities[2]
	features, initStates = getSequenceFiles(activities[2])
	print features
	print initStates


if __name__ == '__main__':
	main()