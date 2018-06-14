from __future__ import division
import numpy as np
import os
import re
import scipy.stats as stats
from collections import deque
import sys
import warnings

GMM_DIM = 64
EPSILON = 0.000001
ITERATIONS = 5
SLF_PROB = 0.9
NXT_PROB = 0.1


HMM_DEF_DICT = "./exc1/hmm_definition.dict"
HMM_DEF_VECT = "./exc1/hmm_definition.vector"
HMM_ST_DEF_VECT = "./exc1/hmm_state_definition.vector"
ST_INIT_SUFFIX = "initStates.npy"
SEQUENCE_LOC = "./exc2/train_samples/"
GRAMMAR_LOC_1 = "./exc1/test1.grammar"
GRAMMAR_LOC_2 = "./exc1/test2.grammar"
GRAMMAR_LOC_3 = "./exc1/test3.grammar"
TEST_DATA_1 = "./exc1/P03_cam01_P03_cereals.npy"
TEST_DATA_2 = "./exc1/P03_cam01_P03_coffee.npy"
TEST_DATA_3 = "./exc1/P03_cam01_P03_milk.npy"
GT_1 = "./exc1/P03_cam01_P03_cereals.gt"
GT_2 = "./exc1/P03_cam01_P03_coffee.gt"
GT_3 = "./exc1/P03_cam01_P03_milk.gt"
GMM_MEANS = "./exc1/GMM_mean.matrix"
GMM_VARS = "./exc1/GMM_var.matrix"



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
	# inf = float("inf")
	arr = arr.copy()
	arr[arr == 0.0] = 1.0
	# arr[arr == inf] = 1.0
	# arr[arr == -inf] = 1.0
	return arr

def sumNormalize(arr):
	"""
	Normalize the array based on sum.
	"""
	s = np.sum(arr)
	if s == 0.0:
		s = 1.0
	return arr / s

def normalizePerRow(arr2d):
	"""
	Ensures that each row sums to 1.0.
	"""
	rowSums = arr2d.sum(axis=1)
	rowSums = fixZeros(rowSums) # prevent division by zero
	# with warnings.catch_warnings():
	# 	warnings.filterwarnings('error')
	# 	try:
	# 		normalized = arr2d / rowSums[:, np.newaxis]
	# 	except Warning as e:
	# 		print rowSums
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
	stateNames = {}
	stateIndices = {}
	stateIndex = 0
	with open(actDefLoc, "r") as actDefFile, open(subActLoc, "r") as subActDefFile:
		for activity in activities:
			stateCount = int(actDefFile.readline().strip())
			stateNameList = []
			stateIndexList = []
			for state in range(stateCount):
				stateNameList.append(subActDefFile.readline().strip())
				stateIndexList.append(stateIndex)
				stateIndex += 1
			stateNames[activity] = stateNameList
			stateIndices[activity] = stateIndexList
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


def getTransitionProbs(stateIndices, selfProb, nxtProb):
	"""
	Constructs the initial state transition probability matrix based on the supplied initial states.
	"""
	nStates = len(stateIndices)
	pTransitions = np.zeros((nStates, nStates), dtype = float) # initialize the N x N state transition matrix.
	# first, the self transitions
	for i in range(nStates):
		for j in range(nStates):
			if j == i:
				prob = selfProb
			elif j == i+1:
				prob = nxtProb
			else:
				prob = 0.0
			pTransitions[i,j] = prob
	pTransitions = normalizePerRow(pTransitions)
	return pTransitions



def getEmissionProbs(stateIndices, obsCount):
	"""
	Constructs the intial observation probability matrix based on supplied initial states and the sequence of observations.
	"""
	pEmissions = np.ones((len(stateIndices), obsCount), dtype = float) # initialize the N x T matrix
	pEmissions = normalizePerRow(pEmissions) # normalize observation probabilities per state
	return pEmissions


def getStateProbs(stateIndices):
	"""
	Constructs the initial state probability matrix. This is just the state distribution.
	"""
	pStates = sumNormalize(np.ones((len(stateIndices), 1), dtype = float))
	return pStates


def compare(stateSeq, gtLoc):
	"""
	Compare the state sequence with the ground truth and return mean-over-frames accuracy.
	"""
	correct = 0
	with open(gtLoc, "r") as gtFile:
		for idx in range(len(stateSeq)):
			actual = gtFile.readline().strip()
			decoded = stateSeq[idx]
			if decoded.startswith(actual):
				correct += 1
	MoF = correct / len(stateSeq)
	return MoF


def pathScore(stateSeq, pEmissions):
	"""
	Get the log likelihood for the given path for the given emission matrix.
	"""
	sumLikelihood = 0
	for obsIdx in range(len(stateSeq)):
		sumLikelihood += pEmissions[stateSeq[obsIdx], obsIdx]
	return sumLikelihood





#############################################################################################



class GrammarContext(object):
	"""
	Holds information for given grammar.
	"""
	def __init__(self, grammarLoc, activities, stateIndicesDict, stateNamesDict):
		super(GrammarContext, self).__init__()
		self.grammarLoc = grammarLoc
		self.activities = activities
		self.stateIndicesDict = stateIndicesDict
		self.stateNamesDict = stateNamesDict
		self.__loadGrammar__()
		self.__makeMappings__()
		self.__flattenNameDict__()

	def __loadGrammar__(self):
		with open(self.grammarLoc, "r") as grammarFile:
			self.activitySeq = grammarFile.readline().strip().split(" ")

	def __makeMappings__(self):
		self.totalStateCount = sum([len(self.stateIndicesDict[activity]) for activity in self.activitySeq])
		self.stateIndexMapping = []
		[self.stateIndexMapping.extend(self.stateIndicesDict[activity]) for activity in self.activitySeq]
	
	def __flattenNameDict__(self):
		self.stateNameMapping = []
		[self.stateNameMapping.extend(self.stateNamesDict[activity]) for activity in self.activities]


	def getTransitionProbs(self, pTransitionsDict):
		"""
		Get the state transition probabilities for the given grammar. 
		"""
		pTransitions = np.zeros((self.totalStateCount, self.totalStateCount), dtype = float)
		start = 0
		for activity in self.activitySeq:
			transMat = pTransitionsDict[activity]
			transMat[-1,-1] = 0 # set the self-transition in last state to zero
			end = start + len(self.stateIndicesDict[activity])
			pTransitions[start:end,start:end] = transMat
			if end < self.totalStateCount:
				pTransitions[end-1,end] = 1.0 # set the transition between activities
			start = end
		return pTransitions


	def getEmissionProbs(self, observations, meanDict, covDict):
		"""
		Get the emission matrix for the given grammar and observations.
		"""
		pEmissions = np.zeros((self.totalStateCount, len(observations)), dtype = float)
		stateIdx = 0
		for activity in self.activitySeq:
			for mu, sigma in zip(meanDict[activity], covDict[activity]):
				pEmissions[stateIdx, :] = stats.multivariate_normal.logpdf(observations, mean = mu, cov = sigma)
				stateIdx += 1
		return pEmissions


	def getStateProbs(self, pStatesDict):
		"""
		Get the state probability matrix.
		"""
		pStates = np.zeros((self.totalStateCount, 1), dtype = float)
		start = 0
		for activity in self.activitySeq:
			stateMat = pStatesDict[activity]
			end = start + len(self.stateIndicesDict[activity])
			pStates[start:end,0] = np.ravel(stateMat)
			start = end
		pStates = sumNormalize(pStates) # must be a probability distribution
		return pStates


	def decodeStateSeq(self, stateSeq):
		"""
		Decode the list of ste indices to state names.
		"""
		return [self.stateNameMapping[self.stateIndexMapping[idx]] for idx in stateSeq]



	


#############################################################################################


class BaumWelch(object):
	"""
	The Baum-Welch algorithm. Parameters are:

	observations: The sequence of observations or feature vectors O(1...T)
	pTransitions: An N x N matrix giving the state transition probabilities. N -> no. of states.
	pEmissions: An N x T- matrix giving the observation probabilities of each of the T features at each state.
	pStates: The probability of each state occurring. N x 1.
	"""
	def __init__(self, activity, observations, pTransitions, pEmissions, pStates):
		super(BaumWelch, self).__init__()
		self.activity = activity
		self.observations = observations
		self.pTransitions = pTransitions
		self.pEmissions = pEmissions
		self.pStates = pStates
		self.stateCount = len(pStates)
		self.obsCount = observations.shape[0] # T
		self.dim = observations.shape[1]


	def run(self):
		# initialize forward and backward passes
		self.alpha = deque()
		self.beta = deque()
		# initialize the temporary variables
		self.gamma = []
		self.eta = []
		self.__temporaries__()
		

	def __passes__(self):
		# initialize the forward and backward passes
		alpha_t = self.pStates * self.pEmissions[:, 0, np.newaxis]
		beta_t = np.ones((self.stateCount, 1), dtype = float)
		self.alpha.append(alpha_t)
		self.beta.append(beta_t)
		for t in range(self.obsCount - 1):
			# perform calculations in components
			c1 = alpha_t * self.pTransitions
			alpha_t = sumNormalize(self.pEmissions[:,t+1].reshape(self.stateCount, 1) * np.sum(c1, axis = 0).reshape(self.stateCount, 1))
			c2 = beta_t * self.pEmissions[:,t+1].reshape(self.stateCount, 1)
			beta_t = sumNormalize(np.sum(self.pTransitions * c2.T, axis = 1)).reshape(self.stateCount, 1)
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
			gamma_t = (self.alpha[t] * self.beta[t]) / divisor
			self.gamma.append(gamma_t)
			if t == self.obsCount-1:
				continue # eta only has T-1 elements 
			beta_t = fixZeros(self.beta[t])
			c1 = gamma_t / beta_t
			c2 = self.pEmissions[:,t+1].reshape(self.stateCount, 1) * self.beta[t+1].reshape(self.stateCount, 1)
			c3 = self.pTransitions * c2.T
			eta_t = c1 * c3
			self.eta.append(eta_t)
		return


	def getTemporaries(self):
		"""
		Return the calculated temporaries.
		"""
		return self.gamma, self.eta



class LearnGaussians(object):
	"""Calculates the parameters of individual Gaussians using the Baum-Welch algorithm."""
	
	def __init__(self, activity, stateNames, stateIndices, seqLoc):
		"""
		Initialize with the activity name and the location of all it's sequences.
		"""
		super(LearnGaussians, self).__init__()
		self.activity = activity
		self.seqFeatures, self.initFiles = getSequenceFiles(activity, seqLoc)
		self.stateNames = stateNames
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
			pTransitions = getTransitionProbs(self.stateIndices, SLF_PROB, NXT_PROB)
			pEmissions = getEmissionProbs(self.stateIndices, len(initStates))
			pStates = getStateProbs(self.stateIndices)
			bw = BaumWelch(self.activity + str(i), observations, pTransitions, pEmissions, pStates)
			self.bwList.append(bw)
			i += 1
		self.dim = self.sequences[0].shape[1]


	def __mu__(self):
		"""
		Get the list of means of all Gaussians.
		"""
		mus = np.zeros((self.stateCount, self.dim), dtype = float)
		sumGamma = np.zeros((self.stateCount, 1), dtype = float)
		for nSeq in range(len(self.seqFeatures)):
			observations = self.sequences[nSeq] # a sequence of T observations
			gamma, _ = self.bwList[nSeq].getTemporaries()
			for t in range(len(gamma)):
				observation_t = observations[t].reshape(self.dim)
				mus += gamma[t] * observation_t
				sumGamma += gamma[t]
		sumGamma = fixZeros(sumGamma)
		mus = mus / sumGamma
		return mus


	def __sigma__(self, mus):
		"""
		Get the list standard deviation of the gaussians.
		"""
		sigmas = np.zeros((self.stateCount, self.dim), dtype = float)
		sumGamma = np.zeros((self.stateCount, 1), dtype = float)
		for nSeq in range(len(self.seqFeatures)):
			observations = self.sequences[nSeq] # a sequence of T observations
			gamma, _ = self.bwList[nSeq].getTemporaries()
			for t in range(len(gamma)):
				observation_t = observations[t].reshape(1, self.dim)
				diff = (observation_t - mus).reshape(mus.shape)
				sigmas += gamma[t] * np.square(diff) # assuma diagonal covariance
				sumGamma += gamma[t]
		sumGamma = fixZeros(sumGamma)
		sigmas = sigmas / sumGamma
		return sigmas


	def __updateTransitionProbs__(self):
		"""
		Update the transition probabilities for the baum-welch algorithm.
		"""
		pTransitions = np.zeros((self.stateCount, self.stateCount), dtype = float)
		sumGamma = np.zeros((self.stateCount, 1), dtype = float)
		for nSeq in range(len(self.seqFeatures)):
			gamma, eta = self.bwList[nSeq].getTemporaries()
			for t in range(len(eta)):
				pTransitions += eta[t]
				sumGamma += gamma[t]
		pTransitions = normalizePerRow(pTransitions)
		for bw in self.bwList:
			bw.pTransitions = pTransitions
		return pTransitions


	def __updateObservationProbs__(self, mus, sigmas):
		"""
		Update the observation probabilities for the baum-welch algorithm.
		"""
		for nSeq in range(len(self.seqFeatures)):
			observations = self.sequences[nSeq] # a sequence of T observations
			pEmissions = np.zeros((self.stateCount, len(observations)))
			for j in range(self.stateCount):
				mu = mus[j]
				sigma = sigmas[j]
				try:
					pEmissions[j, :] = stats.multivariate_normal.pdf(observations, mean = mu, cov = sigma)
				except Exception as e:
					print "mu"
					print mu
					print "sigma"
					print sigma
					raise e
			pEmissions = normalizePerRow(pEmissions)
			self.bwList[nSeq].pEmissions = pEmissions
		return


	def __updateStateProbabilities__(self):
		"""
		Update the initial state distributions.
		"""
		pStates = np.zeros((self.stateCount, 1), dtype = float)
		for nSeq in range(len(self.seqFeatures)):
			gamma, _ = self.bwList[nSeq].getTemporaries()
			pStates += gamma[0]
		pStates = pStates / len(self.seqFeatures)
		pStates = sumNormalize(pStates)
		for bw in self.bwList:
			bw.pStates = pStates
		return pStates


	def __runBaumWelch__(self):
		"""
		run Baum-Welch for next iteration
		"""
		for bw in self.bwList:
			bw.run()
		return


	def run(self):
		"""
		Execute the iterative HMM parameter learning procedure.
		"""
		self.__runBaumWelch__()
		mus = self.__mu__()
		sigmas = self.__sigma__(mus)
		pTransitions = self.__updateTransitionProbs__()
		self.__updateObservationProbs__(mus, sigmas)
		pStates = self.__updateStateProbabilities__()
		return mus, sigmas, pTransitions, pStates


############################################################################################



def viterbi(pStates, pTransitions, pEmissions):
	"""
	Execute the Viterbi algorithm. Returns the sequence of most likely states.
	N -> No. of states
	T -> No. of observations / frames
	Parameters:
	pStates : The initial likelihood of each state. (N x 1)
	pTransitions : The state transition probabilities. (N x N)
	pEmissions : The observation probabilities at each state.
	Contains log-likelihoods instead of likelihoods for numerical stability. (N x T)
	"""
	stateCount = len(pStates) # N
	obsCount = pEmissions.shape[1] # T
	# initialize the tables
	T1 = np.zeros((stateCount, obsCount), dtype = float)
	T2 = np.zeros((stateCount, obsCount), dtype = float)
	for i in range(stateCount):
		T1[i,0] = pStates[i] * pEmissions[i,0]
	
	for t in range(1, obsCount):
		for i in range(stateCount):
			emsn = pEmissions[i,t] # log likelihoods
			probs = T1[:,t-1] + np.ma.log(pTransitions[:,i]).filled(-float('inf')) + emsn
			T1[i,t] = np.max(probs)
			T2[i,t] = np.argmax(probs)

	z = np.full((obsCount), fill_value = -1, dtype = int)
	z[0] = 0
	z[-1] = np.argmax(T1[:,obsCount-1])
	for t in range(obsCount-1, 1, -1):
		z[t-1] = T2[z[t],t]
	return z



############################################################################################


def load():
	"""
	Loads all activities and their corresponding states, the grammars, testa data and the groud truths.
	"""
	activities = getActivities(HMM_DEF_DICT) # load activities
	stateNamesDict, stateIndicesDict = getActivityStates(activities) # load named activity states
	
	# load grammars
	grm1 = GrammarContext(GRAMMAR_LOC_1, activities, stateIndicesDict, stateNamesDict) 
	grm2 = GrammarContext(GRAMMAR_LOC_2, activities, stateIndicesDict, stateNamesDict) 
	grm3 = GrammarContext(GRAMMAR_LOC_3, activities, stateIndicesDict, stateNamesDict)
	grmList = [grm1, grm2, grm3]

	# load test data
	td1 = np.load(TEST_DATA_1).T
	td2 = np.load(TEST_DATA_2).T
	td3 = np.load(TEST_DATA_3).T
	tdList = [td1, td2, td3]

	# load ground truths
	gtList = [GT_1, GT_2, GT_3]

	return activities, stateNamesDict, stateIndicesDict, grmList, tdList, gtList


def test(grmList, tdList, gtList, pTransitionsDict, pStatesDict, meanDict, covDict):
	"""
	Perform inference for all grammars and all test data.
	Return the scores and accuracies per grammar per test data, in a grid.
	"""
	scoreMat = np.zeros((len(grmList), len(tdList)), dtype = float)
	accMat = np.zeros((len(grmList), len(tdList)), dtype = float)
	for i in range(len(grmList)):
		pTransitions = grmList[i].getTransitionProbs(pTransitionsDict)
		pStates = grmList[i].getStateProbs(pStatesDict)
		for j in range(len(tdList)):
			print "Testing for grammar %d, video %d..." % (i, j)
			pEmissions = grmList[i].getEmissionProbs(tdList[j], meanDict, covDict)
			stateIndexSeq = viterbi(pStates, pTransitions, pEmissions)
			stateNameSeq = grmList[i].decodeStateSeq(stateIndexSeq)
			score = pathScore(stateIndexSeq, pEmissions)
			acc = compare(stateNameSeq, gtList[j])
			scoreMat[i,j] = score
			accMat[i,j] = acc
	return scoreMat, accMat


def results(scoreMat, accMat, grmList, tdList):
	"""
	Display results of the experiment.
	"""
	print "SCORE MATRIX (grammar X video)"
	print scoreMat
	print "ACCURACY (MoF) MATRIX (grammar X video)"
	print accMat
	mostLikelyGrammars = np.argmax(scoreMat, axis = 0)
	print "The most likely grammars are: "
	print "For vid1 grammar: %d" % (mostLikelyGrammars[0])	
	print "For vid2 grammar: %d" % (mostLikelyGrammars[1])	
	print "For vid3 grammar: %d" % (mostLikelyGrammars[2])	


def q1():
	"""
	Solution to question 1.
	"""
	# load all data
	activities, stateNamesDict, stateIndicesDict, grmList, tdList, gtList = load()
	
	# load the GMM parameters
	means = np.loadtxt(GMM_MEANS)
	covs = np.loadtxt(GMM_VARS)
	meanDict = {}
	covDict = {}
	pTransitionsDict = {}
	pStatesDict = {}

	# construct data structures per activity
	for activity in activities:
		stateIndices = stateIndicesDict[activity]
		pTransitions = getTransitionProbs(stateIndices, SLF_PROB, NXT_PROB)
		pStates = getStateProbs(stateIndices)
		meanList = [means[i] for i in stateIndices]
		covList = [covs[i] for i in stateIndices]
		meanDict[activity] = meanList
		covDict[activity] = covList
		pTransitionsDict[activity] = pTransitions
		pStatesDict[activity] = pStates
	
	# perform test with the given data and grammars
	scoreMat, accMat = test(grmList, tdList, gtList, pTransitionsDict, pStatesDict, meanDict, covDict)
	results(scoreMat, accMat, grmList, tdList)


def q2():
	activities, stateNamesDict, stateIndicesDict, grmList, tdList, gtList = load()

	# Create a learner list
	lgList = []
	for activity in activities:
		stateNames = stateNamesDict[activity]
		stateIndices = stateIndicesDict[activity]
		lgList.append(LearnGaussians(activity, stateNames, stateIndices, SEQUENCE_LOC))
	
	for it in range(ITERATIONS):
		print "Iteration: %d" % (it)
		meanDict = {}
		covDict = {}
		pTransitionsDict = {}
		pStatesDict = {}
		print "Training..."
		for lg in lgList:
			mus, sigmas, pTransitions, pStates = lg.run()
			meanDict[lg.activity] = mus
			covDict[lg.activity] = sigmas
			pTransitionsDict[lg.activity] = pTransitions
			pStatesDict[lg.activity] = pStates
		scoreMat, accMat = test(grmList, tdList, gtList, pTransitionsDict, pStatesDict, meanDict, covDict)
		print "Testing..."
		results(scoreMat, accMat, grmList, tdList)
	return
	


if __name__ == '__main__':
	# q1() # solution to question 1
	q2() # solution to question 2