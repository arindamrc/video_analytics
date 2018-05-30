import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.externals import joblib
from parameters import *
from utils import *


def combineDescriptors(spatialCsv, temporalCsv):
	"""
	Combine the spatial and temporal video level descriptors obtained after training.
	"""
	headers = ["vidname", "label"]
	for dim in range(VIDEO_DESCRIPTOR_DIM):
		headers.append("dim"+str(dim))
	dfSpatial = pd.read_csv(spatialCsv, names = headers)
	dfTemporal = pd.read_csv(temporalCsv, names = headers)
	# join the two data frames based on video names
	dfMerged = pd.merge(dfSpatial, dfTemporal, on = "vidname", how = "inner", suffixes=("_s", "_t"))
	spatialHeaders = [headers[i] + "_s" for i in range(2, len(headers))]
	temporalHeaders = [headers[i] + "_t" for i in range(2, len(headers))]
	spatialHeaders.extend(temporalHeaders)
	allHeaders = spatialHeaders
	descriptors = dfMerged[allHeaders].values
	labels = dfMerged["label_s"].values
	return descriptors, labels


def main():
	svmTrainData, svmTrainLabels = combineDescriptors(SPATIAL_TRAIN_CSV_LOC, TEMPORAL_TRAIN_CSV_LOC)
	svmTestData, svmTestLabels = combineDescriptors(SPATIAL_TEST_CSV_LOC, TEMPORAL_TEST_CSV_LOC)
	
	# train svm
	linearClassifier = svm.LinearSVC()
	linearClassifier.fit(svmTrainData, svmTrainLabels)
	joblib.dump(linearClassifier, SVM_FILE)

	preds = linearClassifier.predict(svmTestData)
	acc = 0
	for (pred, actual) in zip(preds, svmTestLabels):
		if pred == actual:
			acc = acc + 1
	print "accuracy = %f percent" % ((acc * 100.0)/len(svmTestLabels))

if __name__ == '__main__':
	main()

