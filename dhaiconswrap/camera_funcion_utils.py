import depthai as dhai
import cv2
import numpy as np
import json
try:
	from .pointclouds_utils.pointclouds_manager import PointsCloudManager
	from .calibrationLib.calibration_function import load_calibration_json,check_calibration_exist
except:
	print('PointsCloudManager not loaded within calibration functions')

############################ CONSTANT ###########################################

COLOR_RESOLUTIONS = {"720":dhai.ColorCameraProperties.SensorResolution.THE_720_P,
					 "800":dhai.ColorCameraProperties.SensorResolution.THE_800_P,
					 "1080":dhai.ColorCameraProperties.SensorResolution.THE_1080_P,
					 "4K":dhai.ColorCameraProperties.SensorResolution.THE_4_K,
					 "12MP":dhai.ColorCameraProperties.SensorResolution.THE_12_MP,
					 "13MP":dhai.ColorCameraProperties.SensorResolution.THE_13_MP,}

DEPTH_RESOLUTIONS = {"400":dhai.MonoCameraProperties.SensorResolution.THE_400_P,
					"480":dhai.MonoCameraProperties.SensorResolution.THE_480_P,
					"720":dhai.MonoCameraProperties.SensorResolution.THE_720_P,
					"800":dhai.MonoCameraProperties.SensorResolution.THE_800_P}

MEDIAN_KERNEL = {"0":dhai.MedianFilter.MEDIAN_OFF,
				 "3":dhai.MedianFilter.KERNEL_3x3,
				 "5":dhai.MedianFilter.KERNEL_5x5,
				 "7":dhai.MedianFilter.KERNEL_7x7}


############################ CUSTOM EXCEPTIONS ###########################################

class BlobException(Exception):
	def __init__(self,message):
		super(BlobException,self).__init__(message)

############################ DECORETORS ###########################################

def infoprint(func):
	def wrapper(*args):
		if args[-1]:
			print(f"[INFO]: {func.__name__} Start")
			res = func(*args)
			print(f"[INFO]: {func.__name__} End")
		else:
			res = func(*args)
		return res
	return wrapper

############################ HELP FUNCTIONS ###########################################

def get_available_devices():

	available_devices = dhai.Device.getAllAvailableDevices()
	cameras_id = [device.getMxId() for device in available_devices]
	print(f'[CAMERAS FOUNDED] : {cameras_id}')
	return cameras_id

def create_depthconf_json(path):

	configuration = {"ColorSensorResolution":"1080",
					"StereoSensorResolution":"480",    
					"StereoSensorResolution_calibration":"480",
					"MedianFilterKernel":5,
					"LeftRightCheck":True,
					"ExtendedDisparity": False,
					"Subpixel":False,
					"speckleFilter": False,
					"speckleRange":50,
					"temporalFilter":True,
					"spatialFilter":True,
					"holeFillingRadius":2,
					"numIterations":1,
					"thresholdFilter_minRange":300,
					"thresholdFilter_maxRange":1500,
					"decimationFactor":1,
					"nn_threshold":0.5,
 					"nn_size":[300,300]}
	with open(path,"w") as configfile:
		json.dump(configuration,configfile,indent=1)

	############################ POINTCLOUD MANAGER FUNCTIONS ############################

def create_pointcloud_manager(id=None,calibrationInfo=None,path="./"):

	pointcloud_manager = PointsCloudManager(id)
	if not check_calibration_exist(idname=id,path=path):
		calibrationInfo.append(None)
		pointcloud_manager.SetParameters(calibrationInfo,None,None,True)
		return pointcloud_manager
	calibration_info,roi_2D,viewROI,zdirection = load_calibration_json(path=path,id=id)
	pointcloud_manager.viewROI = viewROI
	calibrationInfo.append(calibration_info)
	pointcloud_manager.SetParameters(calibrationInfo,roi_2D,viewROI,zdirection)

	return pointcloud_manager

############################ HELP CLASS  ###########################################

class IntrinsicParameters():

	def __init__(self,intrinsic_info,size):

		self.fx = intrinsic_info[0][0]
		self.cx = intrinsic_info[0][2]
		self.fy = intrinsic_info[1][1]
		self.cy = intrinsic_info[1][2]
		self.h = size[1]
		self.w = size[0]

	## IMAGE PLOTTING

def visualise_Axes(color_image, calibration_info_devices, len_axes = 0.10):

		
	bounding_box_points_devices = create_axis_draw_points(calibration_info_devices, len_axes)

	points = bounding_box_points_devices.astype(int)
	origin = points[0]

	xend = points[1]
	x_color = (255,0,0)
	cv2.line(color_image, (origin[0],origin[1]), (xend[0],xend[1]), x_color, 2)
	cv2.putText(color_image, "X", (xend[0],xend[1]), cv2.FONT_HERSHEY_PLAIN, 2, x_color )

	yend = points[3]
	y_color = (0,255,0)
	cv2.line(color_image, (origin[0],origin[1]), (yend[0],yend[1]), y_color, 2)
	cv2.putText(color_image, "Y", (yend[0],yend[1]), cv2.FONT_HERSHEY_PLAIN, 2, y_color )

	zend = points[5]
	z_color = (0,0,255)
	cv2.line(color_image, (origin[0],origin[1]), (zend[0],zend[1]), z_color, 2)
	cv2.putText(color_image, "Z", (zend[0],zend[1]), cv2.FONT_HERSHEY_PLAIN, 2, z_color )

	return color_image

def create_draw_points(bounding_box_world_3d, calibration_info_devices):

# Get the bounding box points in the image coordinates
	bounding_box_points_color_image = Convert3Dto2DImage(calibration_info_devices,bounding_box_world_3d.T)

	return bounding_box_points_color_image


def make_axis_points(len_axes):
	x = [0, len_axes, 0, 0, 0, 0]
	y = [0, 0, 0, len_axes, 0, 0]
	z = [0, 0, 0, 0, 0, len_axes]
	return np.stack([x, y, z],1)


def create_axis_draw_points(calibration_info_devices, len_axes):
	bounding_box_world_3d = make_axis_points(len_axes)
	return create_draw_points(bounding_box_world_3d, calibration_info_devices)

def Convert3Dto2DImage(calibration_info,bounding_box_world_3d):

	T0 = calibration_info[3]
	color_intrinsics = calibration_info[0]
	T2 = calibration_info[2] 

	if T0 is not None:
		bounding_box_device_3d = T0.inverse().apply_transformation(bounding_box_world_3d)
		bounding_box_device_3d_RGB = T2.apply_transformation(np.array(bounding_box_device_3d))
		z_RGB = bounding_box_device_3d_RGB[2,:]
		x_RGB = np.divide(bounding_box_device_3d_RGB[0,:],z_RGB)
		y_RGB = np.divide(bounding_box_device_3d_RGB[1,:],z_RGB)
	else:
		bounding_box_device_3d = bounding_box_world_3d
		bounding_box_device_3d_RGB = T2.apply_transformation(bounding_box_device_3d)
		z_RGB = bounding_box_device_3d_RGB[2,:]
		x_RGB = bounding_box_device_3d_RGB[0,:]
		y_RGB = bounding_box_device_3d_RGB[1,:]

	u = (x_RGB * color_intrinsics.fx + color_intrinsics.cx).astype(int)
	v = (y_RGB * color_intrinsics.fy + color_intrinsics.cy).astype(int) 
	
	return np.stack([u,v],0).astype(int).T


