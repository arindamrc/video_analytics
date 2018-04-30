from __future__ import division
import numpy as np
import scipy.ndimage as spnd
import scipy.signal as sig
import cv2
import sys
import os
import sklearn.cluster as skl
from sklearn import svm
from sklearn.externals import joblib

# list of parameters
bin_sep = 90
np_extn = ".npy"
sqrt_s = 2.0 # the multiplier for the variances sigma and tau
codebook_dim = 256
sigma_params = [1,2,3,4,5,6]
tau_params = [1,2]
k = 0.005 # multipler for the trace cubed
kay = 9 # multiplier to find dimension of volume around STIPs.
nx = 3 # no.of cuboids in x-axis
ny = 3 # no.of cuboids in y-axis
nt = 2 # no.of cuboids in t-axis

def getVidAsVol(vidLoc):
	cap = cv2.VideoCapture(vidLoc)

	frame_list = [] #an empty list to hold the frames

	while(cap.isOpened()):
	    ret, frame = cap.read()
	    if not ret:
			print "capture done"
			break	

	    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # get only intensity values
	    frame_list.append(frame)
	    
	cap.release()
	frame_list = np.array(frame_list)
	vidVol = np.stack(frame_list, axis = 2) # stack the frames to get a video volume
	return vidVol


def displayVidVol(vidVol, wait = 10):
	_, _, dp = vidVol.shape
	cv2.namedWindow("VidVol")
	for f in range(dp):
		frame = vidVol[:,:,f]
		cv2.imshow("VidVol", frame)
		cv2.waitKey(wait)
	cv2.destroyAllWindows()


def findMaximaMask(idx_tuple, target):
	for x,y,t in zip(idx_tuple[0], idx_tuple[1], idx_tuple[2]):
		target[x,y,t] = 1
	return target

def normalize(arr):
	norm = np.linalg.norm(arr)
	return arr/norm


def clip(stip, max_x, max_y, max_t):
	stip[0] = min(max_x, stip[0])
	stip[1] = min(max_y, stip[1])
	stip[2] = min(max_t, stip[2])
	stip = np.clip(stip, a_min = 0, a_max = None)
	return stip

def findOpticalFlow(vidVol):
	frame_prev = vidVol[:,:,0]
	_,_,dp = vidVol.shape
	mag = []
	ang = []
	for f in range(1,dp):
		frame = vidVol[:,:,f]
		flow = cv2.calcOpticalFlowFarneback(frame_prev, frame, 0.5, 3, 15, 3, 5, 1.2, 0)
		mag_f, ang_f = cv2.cartToPolar(flow[...,0], flow[...,1])
		mag.append(mag_f)
		ang.append(ang_f)
	
	# there can be only dp - 1 flow frames. adjeust for this by 
	# adding the last flow frame twice.
	mag.append(mag[-1])
	ang.append(ang[-1])
	mag = np.array(mag)
	ang = np.array(ang)
	mag = np.stack(mag, axis = 2)
	ang = np.stack(ang, axis = 2)
	return mag, ang

def makeFeatHistogram(st, en, bins):
	hist = np.array([])
	for bin_i in bins:
		hist_response_i = np.sum(bin_i[st[0]:en[0], st[1]:en[1], st[2]:en[2]])
		hist = np.append(hist, hist_response_i)
	return hist


def findFeatHistContributions(magnitudes, angles):
	"""
	Interpolate and find the contribution of each pixel 
	towards the histogram bins.
	"""
	assert bin_sep < 360
	assert magnitudes.shape == angles.shape
	bin_count = 360 / bin_sep
	if 360 % bin_sep != 0: # ensure uniform bins
		raise ValueError('could not divide bins evenly for bin separation %f' % (bin_sep))
	bin_count = int(bin_count)
	bins = []
	for i in range(bin_count):
		bin_i = angles.copy()
		bin_val = i * bin_sep
		bin_i = 1.0 - (abs(bin_i - bin_val) / bin_sep)

		# discard values not between 0 and 1
		bin_i[abs(bin_i) > 1.0] = 0.0
		bin_i[bin_i < 0.0] = 0.0

		# find histogram response from the magnitude
		bin_i = bin_i * magnitudes
		bins.append(bin_i)
	return bins


def findSTIPs(vidVol):
	vid_desc = [] # initialize video descriptor
	# find flow magnitudes and angles
	stips = np.array([]) # list of STIPs
	# for each scale pair
	for i in sigma_params: 
		for j in tau_params:
			sigma_i = 2 ** ((1 + i) / 2.0)
			tau_j = 2 ** (j / 2.0)

			print "Finding STIPs for sigma = %f, tau = %f" % (sigma_i, tau_j)

			sm_vidVol = spnd.gaussian_filter(vidVol, [sigma_i, sigma_i, tau_j])

			# maybe sobel derivatives?
			Lx = np.gradient(sm_vidVol, axis = 0)
			Ly = np.gradient(sm_vidVol, axis = 1)
			Lt = np.gradient(sm_vidVol, axis = 2)

			# calculate harris matrix components
			Lx2 = Lx * Lx
			Ly2 = Ly * Ly
			Lt2 = Lt * Lt
			LxLy = Lx * Ly
			LyLt = Ly * Lt
			LxLt = Lx * Lt

			# apply re-scaled gaussian kernel for the harris window
			Lx2 = spnd.gaussian_filter(Lx2, [sqrt_s * sigma_i, sqrt_s * sigma_i, sqrt_s * tau_j])
			Ly2 = spnd.gaussian_filter(Ly2, [sqrt_s * sigma_i, sqrt_s * sigma_i, sqrt_s * tau_j])
			Lt2 = spnd.gaussian_filter(Lt2, [sqrt_s * sigma_i, sqrt_s * sigma_i, sqrt_s * tau_j])
			LxLy = spnd.gaussian_filter(LxLy, [sqrt_s * sigma_i, sqrt_s * sigma_i, sqrt_s * tau_j])
			LyLt = spnd.gaussian_filter(LyLt, [sqrt_s * sigma_i, sqrt_s * sigma_i, sqrt_s * tau_j])
			LxLt = spnd.gaussian_filter(LxLt, [sqrt_s * sigma_i, sqrt_s * sigma_i, sqrt_s * tau_j])

			# calculate determinant matrix
			det_mu = (Lx2 * ((Ly2 * Lt2) - (LyLt * LyLt))) - (LxLy * ((LxLy * Lt2) - (LyLt * LxLt))) + (LxLt * ((LxLy * LyLt) - (Ly2 * LxLt)))
			
			# calculate trace and cube it
			trace3_mu = (Lx2 + Ly2 + Lt2) ** 3

			# finally get the corner response matrix
			H = det_mu - (k * trace3_mu)

			# for debug
			# displayVidVol(H, wait = 20)

			# Non-maximum suppression
			# Find maximas along each axis. Then take only those maximas which are present
			# in all three axes.
			idx_maxx = sig.argrelextrema(H, np.greater, axis = 0) 
			idx_maxy = sig.argrelextrema(H, np.greater, axis = 1) 
			idx_maxt = sig.argrelextrema(H, np.greater, axis = 2) 

			H_mask_x = findMaximaMask(idx_maxx, np.zeros_like(H))
			H_mask_y = findMaximaMask(idx_maxy, np.zeros_like(H))
			H_mask_t = findMaximaMask(idx_maxt, np.zeros_like(H))

			# The masks would probably need dilation!
			# multiply the masks element-wise to get local maximas along all axes
			H_mask = H_mask_x * H_mask_y * H_mask_t

			stips_i = np.where(H_mask == 1) # also consider intensity of SPITs

			stips_i = np.stack(stips_i, axis = 1)
			if len(stips) == 0:
				stips = stips_i
			else:
				stips = np.concatenate((stips, stips_i))

			##################################################################################

			# compute HoG and HoF

			# volume dimensions
			del_x = 2 * kay * sigma_i
			del_y = 2 * kay * sigma_i
			del_t = 2 * kay * tau_j

			# pixels per cuboid in each dimension
			px = del_x / nx
			py = del_y / ny
			pt = del_t / nt

			grad_mag = np.sqrt((Lx2 + Ly2)) # the gradient magnitude
			grad_ang = np.arctan2(Ly, Lx) * 180 + np.pi
			grad_ang[grad_ang < 0] = grad_ang[grad_ang < 0] + 360 # the gradient angle (0 to 360 degrees)

			mx, my, mt = grad_mag.shape

			grad_hist_contribs = findFeatHistContributions(grad_mag, grad_ang)

			flow_mag , flow_ang = findOpticalFlow(sm_vidVol)
			flow_hist_contribs = findFeatHistContributions(flow_mag, flow_ang)

			print "STIP count: %d" % (len(stips_i))

			for stip in stips_i: # for each interest point
				# initialize cuboid
				HoG = []
				HoF = []
				cuboid = stip + [ -(nx/2) * px, (ny/2) * py, -(nt/2) * pt] # init to top-left of volume
				for c_x in range(nx):
					cuboid[1] = (ny/2) * py # reset y coord
					for c_y in range(ny):
						cuboid[2] = -(nt/2) * pt # reset t coord
						for c_t in range(nt):
							# find histogram for each cuboid
							st = clip(np.round(cuboid), mx, mx, mt).astype(int) # top-left 
							en = clip(np.round(cuboid + [pt, py, px] + 1), mx, my, mt).astype(int) # bottom-right; numpy array access in non-inclusive

							HoG_cuboid = makeFeatHistogram(st, en, grad_hist_contribs)
							HoF_cuboid = makeFeatHistogram(st, en, flow_hist_contribs)

							HoG.append(HoG_cuboid)
							HoF.append(HoF_cuboid)

							cuboid = cuboid + [0, 0, pt]
						cuboid = cuboid + [0, py, 0]
					cuboid = cuboid + [px, 0, 0]
				
				HoG = np.ndarray.flatten(np.array(HoG))
				HoF = np.ndarray.flatten(np.array(HoF))
				feat_desc = normalize(np.concatenate((HoG, HoF)))
				vid_desc.append(feat_desc)
	vid_desc = np.stack(vid_desc, axis = 1) # combine all STIP descriptors into one 2D array

	return stips, vid_desc


def makeCodebook(vid_descs):
	# vid_descs is a N x 144 dim array; all video descriptors
	vid_descs = np.concatenate(vid_descs)
	print vid_descs.shape
	kmeans = skl.KMeans(n_clusters = codebook_dim, random_state=0, n_jobs = 4).fit(vid_descs)
	return kmeans


def makeVideoHistogram(vid_desc, kmeans):
	bins = {}
	_,feature_dim = vid_desc.shape
	for clust_id in range(codebook_dim):
		bins[clust_id] = 0 # hash centers and initialize the histogram bins
	for feat_desc in vid_desc: # for all features in the video
		feat_desc = feat_desc.reshape(1,feature_dim) # reshape into row vector
		closest = kmeans.predict(feat_desc)[0]
		bins[closest] = bins[closest] + 1
	hist = normalize(np.fromiter(bins.itervalues(), dtype = float))
	return hist


def readLabelsFile(path_data, path_labels):
	space = " "
	samples = []
	end_slash = path_data.endswith("/")
	len_extn = len(".avi")
	with open(path_labels, "r") as f_labels:
		for line in f_labels:
			vid_rel_path,lbl = line.split(space)
			lbl = int(lbl)
			if end_slash:
				vid_path = path_data + vid_rel_path
			else:
				vid_path = path_data + "/" + vid_rel_path
			vid_lbl = vid_rel_path.replace("/", "_")
			vid_lbl = vid_lbl[:-len_extn] # remove the '.avi' extension
			samples.append((vid_path, vid_lbl, lbl))
	return samples



def showSTIPs(vidVol, stips, radius = 1, thickness = 1, wait = 10):
	"""
	Expect the stips sorted in ascending order according to time.
	"""
	stips = stips[stips[:,2].argsort()] # arrange the STIPs in ascending order of time
	ht, wd, dp = vidVol.shape
	cv2.namedWindow("STIPs")	
	for f in range(dp):
		frame = vidVol[:,:,f].copy()
		stips_f = stips[stips[:,2] == f]
		for pt in stips_f:
			cv2.circle(frame, (pt[0], pt[1]), radius, (0,255,255), thickness = thickness)
		cv2.imshow("STIPs", frame)
		cv2.waitKey(wait)
	cv2.destroyAllWindows()


def prepareData(descriptor_loc, samples, codebook):
	data = []
	labels = []
	for sample in samples:
		_, vid_name, labl = sample
		vid_desc = np.load(descriptor_loc + vid_name + np_extn).T
		hist_vid_desc = makeVideoHistogram(vid_desc, codebook)
		hist_vid_desc = hist_vid_desc.reshape(1, codebook_dim)
		data.append(hist_vid_desc)
		labels.append(labl)
	data = np.concatenate(data)
	print "data prepared."
	return data, labels


def extractAndSaveFeatures(descriptor_loc, samples):
	for sample in samples:
		vidLoc, vid_name, _ = sample
		# get the video as a stacked image volume
		vidVol = getVidAsVol(vidLoc)
		print vidVol.shape
		stips, vid_desc = findSTIPs(vidVol)
		# showSTIPs(vidVol, stips, radius = 10, thickness = 2, wait = 30)
		np.save(descriptor_loc + vid_name, vid_desc) # save video descriptor to disk	


def main():
	trn_descLoc = "trn_descriptors/"
	tst_descLoc = "tst_descriptors/"
	f_kmeans = "codebook.pkl"
	f_svm = "classifier.pkl"
	if not os.path.exists(trn_descLoc):
		os.makedirs(trn_descLoc)
	if not os.path.exists(tst_descLoc):
		os.makedirs(tst_descLoc)

	train_data = readLabelsFile("/home/arc/VA_Assignments/Datasets/Wiezmann", "/home/arc/VA_Assignments/Datasets/Wiezmann/train.txt")
		
	extractAndSaveFeatures(trn_descLoc, train_data)

	test_data = readLabelsFile("/home/arc/VA_Assignments/Datasets/Wiezmann", "/home/arc/VA_Assignments/Datasets/Wiezmann/test.txt")

	extractAndSaveFeatures(tst_descLoc, test_data)
	
	all_vids_desc = []
	for train_sample in train_data:
		_, vid_name, _ = train_sample
		vid_desc = np.load(trn_descLoc + vid_name + np_extn).T
		all_vids_desc.append(vid_desc)

	print "making codebook..."
	codebook = makeCodebook(all_vids_desc)
	joblib.dump(codebook, f_kmeans)
	print "codebook made."

	# loading codebook
	# codebook = joblib.load(f_kmeans)

	print "commencing training..."
	svm_trn_data, svm_trn_labl = prepareData(trn_descLoc, train_data, codebook)

	# classify using linear SVM
	lin_clf = svm.LinearSVC()
	lin_clf.fit(svm_trn_data, svm_trn_labl)
	joblib.dump(lin_clf, f_svm)
	print "trained."

	# lin_clf = joblib.load(f_svm)

	print "testing..."
	svm_tst_data, svm_tst_labl = prepareData(tst_descLoc, test_data, codebook)

	preds = lin_clf.predict(svm_tst_data)
	acc = 0
	for (pred, actual) in zip(preds, svm_tst_labl):
		if pred == actual:
			acc = acc + 1
	print "accuracy = %f percent" % ((acc * 100.0)/len(svm_tst_labl))

if __name__ == '__main__':
	main()