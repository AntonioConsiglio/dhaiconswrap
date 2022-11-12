import os
import json
import numpy as np
import cv2
from .calibration_kabsch import PoseEstimation, Transformation

def docalibration(device_manager, chessboard_params, calibration_roi, shiftcalibration,zdirection = False,verbose=False, path = "./"):
	'''
	#This function is used to calibrate one or more cameras in 3D space.\n
	input:\n
	- device_manager : class to manage cameras\n
	- chessboard_params: [n_corners_along_h, n_corners_along_w, square_size]\n
	- calibration_roi: Region of interest\n
	- shiftcalibration: a vector if there is a shift of the calibration\n
	- zdirection: bool, if True the the Z cordinate will be positive if over the chessboard plane else negative\n
	- verbose: bool, if True a chessboard image with corners colored in green will be saved in home path\n
	- path: the path where the .json file with calibration parameters (trasportation matrix) is stored\n
	'''
	# Set the chessboard parameters for calibration 
	# Estimate the pose of the chessboard in the world coordinate using the Kabsch Method
	try:
		calibrated_device_count = 0
		intrinsics_devices,extriniscs_device = device_manager.get_intrisic_and_extrinsic()
		while calibrated_device_count < 1: #len(device_manager._available_devices)
			state,frames,_ = device_manager.pull_for_frames()
			if state:
				pose_estimator = PoseEstimation(frames, intrinsics_devices,extriniscs_device, chessboard_params)
				if device_manager.verbose or verbose:
					transformation_result_kabsch, corners3D,chessboard_image = pose_estimator.perform_pose_estimation(True)
					cv2.imwrite("chessboard_corners_finded.png",chessboard_image)
				else:
					transformation_result_kabsch, corners3D,_ = pose_estimator.perform_pose_estimation()
				#object_point, _ = pose_estimator.get_chessboard_corners_in3d()
				calibrated_device_count = 0

				if not transformation_result_kabsch[0] and transformation_result_kabsch[1] is None:
					print("Place the chessboard on the plane where the object needs to be detected..")
				elif transformation_result_kabsch[0]:
					calibrated_device_count += 1

		# Save the transformation object for all devices in an array to use for measurements

		transformation_device= transformation_result_kabsch[1].inverse()
		t = Transformation(translation_vector = -np.array(shiftcalibration))
		mf = np.dot(t.get_matrix(),transformation_device.get_matrix())
		if zdirection:
			rotationmatrix = np.array([[1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,1]])
			mf = np.dot(rotationmatrix,mf)
		transformation_device.set_matrix(mf)

		roi_2D = calibration_roi

		save_calibration_json(transformation_device, roi_2D,zdirection, path)

		return corners3D
	except Exception as e:
		print(e)

def domulticalibration(device_managers, chessboard_params, calibration_roi, shiftcalibration,zdirection = False,verbose=False,path = "./"):
	'''
	#This function is used to calibrate more then one cameras in 3D space.\n
	input:\n
	- device_manager : dictionary within class to manage cameras\n
	- chessboard_params: [n_corners_along_h, n_corners_along_w, square_size]\n
	- calibration_roi: Region of interest\n
	- shiftcalibration: a vector if there is a shift of the calibration\n
	- zdirection: bool, if True the the Z cordinate will be positive if over the chessboard plane else negative\n
	- verbose: bool, if True a chessboard image with corners colored in green will be saved in home path\n
	- path: the path where the .json file with calibration parameters (trasportation matrix) is stored\n
	'''

	try:
		calibrated_device_count = 0
		intrinsics_devices = {}
		extrinsics_devices = {}
		transformation_results_kabsch = {}
		for key, device in device_managers.items():
			intrinsics_devices[key],extrinsics_devices[key] = device.get_intrisic_and_extrinsic()
		while calibrated_device_count < len(device_managers):
			calibrated_device_count = 0
			for key, device in device_managers.items():

				state,frames,_ = device.pull_for_frames()
				if state:
					pose_estimator = PoseEstimation(frames, intrinsics_devices[key],extrinsics_devices[key], chessboard_params)
					print(f"[{key}] : PERFORM POSE ESTIMATION")
					if device.verbose or verbose:
						transformation_results_kabsch[key], corners3D,chessboard_image = pose_estimator.perform_pose_estimation(True)
						cv2.imwrite(f"chessboard_corners_finded_{key}.png",chessboard_image)
					else:
						transformation_results_kabsch[key], corners3D,_ = pose_estimator.perform_pose_estimation()
					#object_point, _ = pose_estimator.get_chessboard_corners_in3d()

					if not transformation_results_kabsch[key][0] and transformation_results_kabsch[key][1] is None:
						print(f"For DEVICE: {key} ==>")
						print("Place the chessboard on the plane where the object needs to be detected..")
					elif transformation_results_kabsch[key][0]:
						calibrated_device_count += 1

		# Save the transformation object for all devices in an array to use for measurements
		cv2.destroyAllWindows()
		transformation_devices = {}
		for key,transformation_result_kabsch in transformation_results_kabsch.items():
			transformation_device= transformation_result_kabsch[1].inverse()
			t = Transformation(translation_vector = -np.array(shiftcalibration))
			mf = np.dot(t.get_matrix(),transformation_device.get_matrix())
			if zdirection:
				rotationmatrix = np.array([[1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,1]])
				mf = np.dot(rotationmatrix,mf)
			transformation_device.set_matrix(mf)
			transformation_devices[key] = transformation_device

		roi_2D = calibration_roi

		save_calibration_json(transformation_devices, roi_2D,zdirection, path)
		return corners3D
	except Exception as e:
		print(e)

def check_calibration_exist(idname,path = "./"):

	if not os.path.isfile(get_roi_2D_name(path)):
		return False
	if not os.path.isfile(get_calibration_name(path,idname)):
		return False
	return True

def load_calibration_json(path = "./",id=None):

	if id is None:
		with open(os.path.join(path,'camera_calibration.json'),"r") as json_file:
			file = json.load(json_file)
			pose_mat = np.array(file['extrinsic_matrix'])
			transformation_devices = Transformation(pose_mat[:3,:3],pose_mat[:3,3])
			zdirection = file['zdirection']
	else:
		with open(os.path.join(path,f'{id}_camera_calibration.json'),'r') as json_file:
			file = json.load(json_file)
			pose_mat = np.array(file['extrinsic_matrix'])
			transformation_devices = Transformation(pose_mat[:3,:3],pose_mat[:3,3])
			zdirection = file['zdirection']


	with open(os.path.join(path,'roi_2D.json'),"r") as json_file:
		roi_2D = json.load(json_file)
		
	viewROI = roi_2D

	return transformation_devices, roi_2D, viewROI,zdirection

def get_calibration_name(path,id=None):
	if id is None:
		return os.path.join(path,'camera_calibration.json')
	else:
		return os.path.join(path,f'{id}_camera_calibration.json')

def get_roi_2D_name(path):
	return os.path.join(path, 'roi_2D.json')

def save_calibration_json(transformation_devices, roi_2D, zdirection, path = "./"):
	
	if type(transformation_devices) is dict:
		for key,transformation_device in transformation_devices.items():
			mat = transformation_device.get_matrix()
			towrite = {"extrinsic_matrix":mat.tolist(),"zdirection":zdirection}
			with open(get_calibration_name(path,key), 'w') as outfile:
				json.dump(towrite, outfile)
	else:
		transformation_device = transformation_devices
		mat = transformation_device.get_matrix()
		towrite = {"extrinsic_matrix":mat.tolist(),"zdirection":zdirection}
		with open(get_calibration_name(path), 'w') as outfile:
			json.dump(towrite, outfile)
	with open(get_roi_2D_name(path), 'w') as outfile:
		json.dump(roi_2D, outfile)

