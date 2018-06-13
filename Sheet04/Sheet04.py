from __future__ import division
import numpy as np
import os
import re
import scipy.stats as stats
from collections import deque
import sys

GMM_DIM = 64
EPSILON = 0.000001
ITERATIONS = 5
SLF_PROB = 0.9
NXT_PROB = 0.1


HMM_DEF_DICT = "/home/arc/VA_Assignments/Sheet04/exc1/hmm_definition.dict"
HMM_DEF_VECT = "/home/arc/VA_Assignments/Sheet04/exc1/hmm_definition.vector"
HMM_ST_DEF_VECT = "/home/arc/VA_Assignments/Sheet04/exc1/hmm_state_definition.vector"
ST_INIT_SUFFIX = "initStates.npy"
SEQUENCE_LOC = "/home/arc/VA_Assignments/Sheet04/exc2/train_samples/"
GRAMMAR_LOC_1 = "/home/arc/VA_Assignments/Sheet04/exc1/test1.grammar"
GRAMMAR_LOC_2 = "/home/arc/VA_Assignments/Sheet04/exc1/test2.grammar"
GRAMMAR_LOC_3 = "/home/arc/VA_Assignments/Sheet04/exc1/test3.grammar"
TEST_DATA_1 = "/home/arc/VA_Assignments/Sheet04/exc1/P03_cam01_P03_cereals.npy"
TEST_DATA_2 = "/home/arc/VA_Assignments/Sheet04/exc1/P03_cam01_P03_coffee.npy"
TEST_DATA_3 = "/home/arc/VA_Assignments/Sheet04/exc1/P03_cam01_P03_milk.npy"
GT_1 = "/home/arc/VA_Assignments/Sheet04/exc1/P03_cam01_P03_cereals.gt"
GT_2 = "/home/arc/VA_Assignments/Sheet04/exc1/P03_cam01_P03_coffee.gt"
GT_3 = "/home/arc/VA_Assignments/Sheet04/exc1/P03_cam01_P03_milk.gt"
GMM_MEANS = "/home/arc/VA_Assignments/Sheet04/exc1/GMM_mean.matrix"
GMM_VARS = "/home/arc/VA_Assignments/Sheet04/exc1/GMM_var.matrix"



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
	arr[arr == 0.0] = 1.0
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
			actual = gtFile.readline()
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
		alpha_t = self.pStates * self.pEmissions[:, 0, np.newaxis]
		beta_t = np.ones((self.stateCount, 1), dtype = float)
		self.alpha.append(alpha_t)
		self.beta.append(beta_t)
		for t in range(self.obsCount - 1):
			for i in range(self.stateCount):
				sumAlpha = 0
				sumBeta = 0
				for j in range(self.stateCount):
					sumAlpha += alpha_t[j] * self.pTransitions[j,i]
					sumBeta += beta_t[j] * self.pTransitions[i,j] * self.pEmissions[j,(self.obsCount - 1 - t)]
				alpha_t[i] = self.pEmissions[i, t+1] * sumAlpha
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
					eta_t[i,j] = self.gamma[t][i] * self.pTransitions[i,j] * self.pEmissions[j, t+1] * self.beta[t+1][j] / beta_t[i]
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
		pTransitions = normalizePerRow(pTransitions, checkZeros = False)
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
				singular = False
				if abs(np.linalg.det(sigma)) == 0:
					pEmissions[j, :] = np.zeros((len(observations)), dtype = float) # singular sigma
				else:
					pEmissions[j, :] = stats.multivariate_normal.pdf(observations, mean = mu, cov = sigma)
			pEmissions = normalizePerRow(pEmissions, checkZeros = False)
			self.bwList[nSeq].pEmissions = pEmissions
		return


	def __updateStateProbabilities__(self):
		"""
		Update the initial state distributions.
		"""
		pStates = np.zeros((len(self.stateCount), 1), dtype = float)
		for nSeq in range(len(self.seqFeatures)):
			gamma, _ = self.bwList[nSeq].getTemporaries()
			pStates += gamma[0]
		pStates = pStates / len(self.seqFeatures)
		pStates = sumNormalize(pStates)
		for bw in self.bwList:
			bw.pStates = pStates
		return pStates


	def __resetBaumWelch__(self):
		"""
		Reset Baum-Welch for next iteration
		"""
		for bw in self.bwList:
			bw.reset()
		return


	def run(self):
		"""
		Execute the iterative HMM parameter learning procedure.
		"""
		mus = self.__mu__()
		sigmas = self.__sigma__(mus)
		pTransitions = self.__updateTransitionProbs__()
		self.__updateObservationProbs__(mus, sigmas)
		pStates = self.__updateStateProbabilities__()
		self.__resetBaumWelch__()
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
			probs = T1[:,t-1] + np.ma.log(pTransitions[:,i]).filled(0) + emsn
			T1[i,t] = np.max(probs)
			T2[i,t] = np.argmax(probs)

	z = deque()
	z.append(np.argmax(T1[:,obsCount-1]))
	for t in range(2, obsCount-1, -1):
		z.appendleft(T2[z[t],t])
	return z



############################################################################################


def load():
	activities = getActivities(HMM_DEF_DICT)
	stateNamesDict, stateIndicesDict = getActivityStates(activities)
	grm1 = GrammarContext(GRAMMAR_LOC_1, activities, stateIndicesDict, stateNamesDict) 
	grm2 = GrammarContext(GRAMMAR_LOC_2, activities, stateIndicesDict, stateNamesDict) 
	grm3 = GrammarContext(GRAMMAR_LOC_3, activities, stateIndicesDict, stateNamesDict)
	grmList = [grm1, grm2, grm3]

	td1 = np.load(TEST_DATA_1).T
	td2 = np.load(TEST_DATA_2).T
	td3 = np.load(TEST_DATA_3).T
	tdList = [td1, td2, td3]

	gtList = [GT_1, GT_2, GT_3]

	return activities, stateNamesDict, stateIndicesDict, grmList, tdList, gtList


def test(grmList, tdList, gtList, pTransitionsDict, pStatesDict, meanDict, covDict):
	scoreMat = np.zeros((len(grmList), len(tdList)), dtype = float)
	accMat = np.zeros((len(grmList), len(tdList)), dtype = float)
	for i in range(len(grmList)):
		pTransitions = grmList[i].getTransitionProbs(pTransitionsDict)
		pStates = grmList[i].getStateProbs(pStatesDict)
		for j in range(len(tdList)):
			pEmissions = grmList[i].getEmissionProbs(tdList[j], meanDict, covDict)
			stateIndexSeq = viterbi(pStates, pTransitions, pEmissions)
			stateNameSeq = grmList[i].decodeStateSeq(stateIndexSeq)
			score = pathScore(stateIndexSeq, pEmissions)
			acc = compare(stateNameSeq, gtList[j])
			scoreMat[i,j] = score
			accMat[i,j] = acc
	return scoreMat, accMat


def q1():
	activities, stateNamesDict, stateIndicesDict, grmList, tdList, gtList = load()
	means = np.loadtxt(GMM_MEANS)
	covs = np.loadtxt(GMM_VARS)
	meanDict = {}
	covDict = {}
	pTransitionsDict = {}
	pStatesDict = {}
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
	scoreMat, accMat = test(grmList, tdList, gtList, pTransitionsDict, pStatesDict, meanDict, covDict)
	print "scoreMat"
	print scoreMat
	print "accMat"
	print accMat


def q2():
	activities, stateNamesDict, stateIndicesDict, grmList, tdList, gtList = load()

	scoreMat = np.zeros((len(grmList), len(tdList), ITERATIONS), dtype = float)
	accMat = np.zeros((len(grmList), len(tdList), ITERATIONS), dtype = float)

	lgList = []
	for activity in activities:
		stateNames = stateNamesDict[activity]
		stateIndices = stateIndicesDict[activity]
		lgList.append(LearnGaussians(activity, stateNames, stateIndices, SEQUENCE_LOC))
	
	for it in range(ITERATIONS):
		meanDict = {}
		covDict = {}
		pTransitionsDict = {}
		pStatesDict = {}
		for lg in lgList:
			mus, sigmas, pTransitions, pStates = lg.run()
			meanDict[activity] = mus
			covDict[activity] = sigmas
			pTransitionsDict[activity] = pTransitions
			pStatesDict[activity] = pStates
		scr, acc = test(grmList, tdList, gtList, pTransitionsDict, pStatesDict, meanDict, covDict)
		scoreMat[:,:,it] = scr
		accMat[:,:,it] = acc

	# for it in range(ITERATIONS):
	# 	mostLikelyGrammars = np.argmax(scoreMat[:,:,it], axis = 0)
	# 	print "for iteration %d, the most likely grammars are: " % (it)
	# 	print "vid1 : %d" % (mostLikelyGrammars[0])	
	# 	print "vid2 : %d" % (mostLikelyGrammars[1])	
	# 	print "vid3 : %d" % (mostLikelyGrammars[2])	

	print "scoreMat"
	print scoreMat
	print "accMat"
	print accMat

	


if __name__ == '__main__':
	q1()
	# q2()