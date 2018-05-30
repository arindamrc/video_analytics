# Some parameters
VIDEO_FRAME_SAMPLE_RATE = 10
CONVERT = True
VIDEO_INPUT_FRAME_COUNT = 3
VIDEO_INPUT_FLOW_COUNT = 5
TRAIN_BATCH_SIZE = 64
NWORKERS_LOADER = 4 
SHUFFLE_LOADER = True
CROP_SIZE_TF = 224 
HORIZONTAL_FLIP_TF = True 
NORM_MEANS_TF = [0.485, 0.456, 0.406] 
NORM_STDS_TF = [0.229, 0.224, 0.225]
NACTION_CLASSES = 101
NEPOCHS = 20
INITIAL_LR = 0.001
MOMENTUM_VAL = 0.9
MILESTONES_LR = [1,2]
VIDEO_DESCRIPTOR_DIM = 256

# Some constants
VIDEO_EXTN = ".avi"
FRAME_EXTN = ".jpg"
DATA_DIR = "/media/data/fmthoker/mini-UCF-101"
FLOW_DATA_DIR = "/media/data/fmthoker/mini-ucf101_flow_img_tvl1_gpu"
FRAMES_DIR_TRAIN = "/media/remote_home/va06/VA/Sheet03/mini-UCF-101-frames-train"
FRAMES_DIR_TEST = "/media/remote_home/va06/VA/Sheet03/mini-UCF-101-frames-test"
VIDEOLIST_TRAIN = "/media/remote_home/va06/VA/Sheet03/demoTrain.txt"
VIDEOLIST_TEST = "/media/remote_home/va06/VA/Sheet03/demoTest.txt"
ACTIONLABEL_FILE = "/media/data/fmthoker/mini-UCF-101/ucfTrainTestlist/classInd.txt"
CHECKPOINT_DIR = "/media/remote_home/va06/VA/Sheet03/checkpoints/"
SPATIAL_CKP_FILE = "spatial_ckp.pth.tar"
SPATIAL_BEST_FILE = "spatial_best.pth.tar"
MOTION_CKP_FILE = "temporal_ckp.pth.tar"
MOTION_BEST_FILE = "temporal_best.pth.tar"
X_PREFIX_FLOW = "flow_x_"
Y_PREFIX_FLOW = "flow_y_"
TEMPORAL_TRAIN_CSV_LOC = "/media/remote_home/va06/VA/Sheet03/temporal_train_desc.csv"
TEMPORAL_TEST_CSV_LOC = "/media/remote_home/va06/VA/Sheet03/temporal_test_desc.csv"
SPATIAL_TEST_CSV_LOC = "/media/remote_home/va06/VA/Sheet03/spatial_test_desc.csv"
SPATIAL_TRAIN_CSV_LOC = "/media/remote_home/va06/VA/Sheet03/spatial_train_desc.csv"
SPATIAL_PERFORMANCE_LOC = "/media/remote_home/va06/VA/Sheet03/spatial_performance.csv"
TEMPORAL_PERFORMANCE_LOC = "/media/remote_home/va06/VA/Sheet03/temporal_performance.csv"