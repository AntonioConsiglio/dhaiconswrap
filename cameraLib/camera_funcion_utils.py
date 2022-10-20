import depthai as dhai
try:
	from .pointclouds_utils.pointclouds_manager import PointsCloudManager
	from .calibrationLib.calibration_function import load_calibration_json,check_calibration_exist
except:
	print('PointsCloudManager not loaded within calibration functions')

############################ RGB SENSOR CONFIGURATION FUNCTIONS ############################

def configure_rgb_sensor(pipeline,size,fps,nn_active,blob_path,calibration=False):

		cam_rgb = pipeline.create(dhai.node.ColorCamera)
		cam_rgb.setResolution(dhai.ColorCameraProperties.SensorResolution.THE_1080_P)
		if calibration:
			pass
			#cam_rgb.initialControl.setManualFocus(130)
		cam_rgb.setPreviewSize(size)
		cam_rgb.setBoardSocket(dhai.CameraBoardSocket.RGB)
		cam_rgb.setInterleaved(False)
		cam_rgb.setFps(fps)
		xout_rgb = pipeline.create(dhai.node.XLinkOut)
		xout_rgb.setStreamName("rgb")
		cam_rgb.preview.link(xout_rgb.input)
		if nn_active:
			manip,_= configure_image_manipulator(pipeline)
			cam_rgb.preview.link(manip.inputImage)
			configure_nn_node(manip,pipeline,blob_path)
		return xout_rgb

def configure_image_manipulator(pipeline):

    manip = pipeline.create(dhai.node.ImageManip)
    manipOut = pipeline.create(dhai.node.XLinkOut)
    manipOut.setStreamName('neural_input')
    manip.initialConfig.setResize(300,300)
    manip.initialConfig.setFrameType(dhai.ImgFrame.Type.BGR888p)
    manip.out.link(manipOut.input)
    
    return manip,manipOut

def configure_nn_node(manip,pipeline,blob_path):
		
		nn = pipeline.create(dhai.node.MobileNetDetectionNetwork)
		nnOut = pipeline.create(dhai.node.XLinkOut)
		nnOut.setStreamName("neural")
		# define nn features
		nn.setConfidenceThreshold(0.5)
		nn.setBlobPath(blob_path)
		nn.setNumInferenceThreads(2)
		# Linking
		manip.out.link(nn.input)
		nn.out.link(nnOut.input)

############################ DEPTH CONFIGURATION FUNCTIONS ############################

def configure_depth_sensor(pipeline,calibration):

    monoLeft = pipeline.create(dhai.node.MonoCamera)
    monoRight = pipeline.create(dhai.node.MonoCamera)
    depth = pipeline.create(dhai.node.StereoDepth)
    xout_depth = pipeline.create(dhai.node.XLinkOut)
    xout_depth.setStreamName("depth")
    xout_disparity = pipeline.create(dhai.node.XLinkOut)
    xout_disparity.setStreamName("disparity")
    configure_depth_proprieties(monoLeft,monoRight,depth,calibration)
    if calibration:
        depth.setDepthAlign(dhai.CameraBoardSocket.RGB)
    monoLeft.out.link(depth.left)
    monoRight.out.link(depth.right)
    depth.disparity.link(xout_disparity.input)
    depth.depth.link(xout_depth.input)

def configure_depth_proprieties(left,right,depth,calibration):
	if not calibration:
		left.setResolution(dhai.MonoCameraProperties.SensorResolution.THE_480_P)
		left.setBoardSocket(dhai.CameraBoardSocket.LEFT)
		right.setResolution(dhai.MonoCameraProperties.SensorResolution.THE_480_P)
		right.setBoardSocket(dhai.CameraBoardSocket.RIGHT)
	else:
		left.setResolution(dhai.MonoCameraProperties.SensorResolution.THE_480_P)
		left.setBoardSocket(dhai.CameraBoardSocket.LEFT)
		right.setResolution(dhai.MonoCameraProperties.SensorResolution.THE_480_P)
		right.setBoardSocket(dhai.CameraBoardSocket.RIGHT)


	# Create a node that will produce the depth map (using disparity output as it's easier to visualize depth this way)
	depth.setDefaultProfilePreset(dhai.node.StereoDepth.PresetMode.HIGH_DENSITY)
	# Options: MEDIAN_OFF, KERNEL_3x3, KERNEL_5x5, KERNEL_7x7 (default)
	depth.initialConfig.setMedianFilter(dhai.MedianFilter.KERNEL_5x5)
	depth.setLeftRightCheck(True)
	depth.setExtendedDisparity(True)
	depth.setSubpixel(False)

	config = depth.initialConfig.get()
	config.postProcessing.speckleFilter.enable = False
	config.postProcessing.speckleFilter.speckleRange = 50
	config.postProcessing.temporalFilter.enable = True
	config.postProcessing.spatialFilter.enable = True
	config.postProcessing.spatialFilter.holeFillingRadius = 2
	config.postProcessing.spatialFilter.numIterations = 1
	config.postProcessing.thresholdFilter.minRange = 300
	config.postProcessing.thresholdFilter.maxRange = 650
	config.postProcessing.decimationFilter.decimationFactor = 1
	depth.initialConfig.set(config)

	############################ POINTCLOUD MANAGER FUNCTIONS ############################

def create_pointcloud_manager(id,calibrationInfo):

	pointcloud_manager = PointsCloudManager(id)
	check_calibration_exist()
	calibration_info,roi_2D,viewROI = load_calibration_json()
	pointcloud_manager.viewROI = viewROI
	calibrationInfo.append(calibration_info)
	pointcloud_manager.SetParameters(calibrationInfo,roi_2D,viewROI)

	return pointcloud_manager