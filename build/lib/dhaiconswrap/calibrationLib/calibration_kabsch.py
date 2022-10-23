##################################################################################################
##       License: Apache 2.0. See LICENSE file in root directory.		                      ####
##################################################################################################
##                  Box Dimensioner with multiple cameras: Helper files 					  ####
##################################################################################################

from . import calculate_rmsd_kabsch as rmsd
import numpy as np
from .helper_functions import cv_find_chessboard, get_chessboard_points_3D, get_depth_at_pixel, convert_depth_pixel_to_metric_coordinate
import cv2

# """
#   _   _        _                      _____                     _    _                    
#  | | | |  ___ | | _ __    ___  _ __  |  ___|_   _  _ __    ___ | |_ (_)  ___   _ __   ___ 
#  | |_| | / _ \| || '_ \  / _ \| '__| | |_  | | | || '_ \  / __|| __|| | / _ \ | '_ \ / __|
#  |  _  ||  __/| || |_) ||  __/| |    |  _| | |_| || | | || (__ | |_ | || (_) || | | |\__ \
#  |_| |_| \___||_|| .__/  \___||_|    |_|    \__,_||_| |_| \___| \__||_| \___/ |_| |_||___/
# 				 _|                                                                      
# """


def calculate_transformation_kabsch(src_points, dst_points):
	"""
	Calculates the optimal rigid transformation from src_points to
	dst_points
	(regarding the least squares error)

	Parameters:
	-----------
	src_points: array
		(3,N) matrix
	dst_points: array
		(3,N) matrix
	
	Returns:
	-----------
	rotation_matrix: array
		(3,3) matrix
	
	translation_vector: array
		(3,1) matrix

	rmsd_value: float

	"""
	assert src_points.shape == dst_points.shape
	if src_points.shape[0] != 3:
		raise Exception("The input data matrix had to be transposed in order to compute transformation.")
		
	src_points = src_points.transpose()
	dst_points = dst_points.transpose()
	
	src_points_centered = src_points - rmsd.centroid(src_points)
	dst_points_centered = dst_points - rmsd.centroid(dst_points)

	rotation_matrix = rmsd.kabsch(src_points_centered, dst_points_centered)
	rmsd_value = rmsd.kabsch_rmsd(src_points_centered, dst_points_centered)

	translation_vector = rmsd.centroid(dst_points) - np.matmul(rmsd.centroid(src_points), rotation_matrix)

	return rotation_matrix.transpose(), translation_vector.transpose(), rmsd_value



# """
#   __  __         _           ____               _                _   
#  |  \/  |  __ _ (_) _ __    / ___| ___   _ __  | |_  ___  _ __  | |_ 
#  | |\/| | / _` || || '_ \  | |    / _ \ | '_ \ | __|/ _ \| '_ \ | __|
#  | |  | || (_| || || | | | | |___| (_) || | | || |_|  __/| | | || |_ 
#  |_|  |_| \__,_||_||_| |_|  \____|\___/ |_| |_| \__|\___||_| |_| \__|																	 

# """

class Transformation:
	def __init__(self, rotation_matrix = None, translation_vector = None, trasformation_mat = None):

		if trasformation_mat is None:
			self.pose_mat = np.zeros((4,4))

			if rotation_matrix is not None:
				self.pose_mat[:3,:3] = rotation_matrix
			else:
				self.pose_mat[:3,:3] = np.eye(3)

			if translation_vector is not None:
				self.pose_mat[:3,3] = translation_vector.flatten()

			self.pose_mat[3,3] = 1
		else:
			self.pose_mat = trasformation_mat

	def get_matrix(self):
		return self.pose_mat

	def set_matrix(self, pose_mat):
		self.pose_mat = pose_mat
	
	def apply_transformation(self, points):
		""" 
		Applies the transformation to the pointcloud
		
		Parameters:
		-----------
		points : array
			(3, N) matrix where N is the number of points
		
		Returns:
		----------
		points_transformed : array
			(3, N) transformed matrix
		"""
		assert(points.shape[0] == 3)
		n = points.shape[1] 
		points_ = np.vstack((points, np.ones((1,n))))
		points_trans_ = np.matmul(self.pose_mat, points_)
		points_transformed = np.true_divide(points_trans_[:3,:], points_trans_[[-1], :])
		return points_transformed
	
	def inverse(self):
		"""
		Computes the inverse transformation and returns a new Transformation object

		Returns:
		-----------
		inverse: Transformation

		"""
		rotation_matrix = self.pose_mat[:3,:3]
		translation_vector = self.pose_mat[:3,3]
		
		rot = np.transpose(rotation_matrix)
		trans = - np.matmul(np.transpose(rotation_matrix), translation_vector)
		return Transformation(rot, trans)
	
	

class PoseEstimation:

	def __init__(self, frames, intrinsic,extrinsic,chessboard_params):
		assert(len(chessboard_params) == 3)
		self.frames = frames
		self.intrinsic = intrinsic
		self.deep_to_rgb_extrinsic = extrinsic
		self.chessboard_params = chessboard_params		

	def get_chessboard_corners_in3d(self):
		"""
		Searches the chessboard corners in the grayscale images of 
		the connected device and uses the information in the 
		corresponding depth image to calculate the 3d 
		coordinates of the chessboard corners in the coordinate system of 
		the camera

		Returns:
		-----------
		corners3D : 
			values: [success, points3D, validDepths] 
				success: bool
					Indicates wether the operation was successfull
				points3d: array
					(3,N) matrix with the coordinates of the chessboard corners
					in the coordinate system of the camera. N is the number of corners
					in the chessboard. May contain points with invalid depth values
				validDephts: [bool]*
					Sequence with length N indicating which point in points3D has a valid depth value
		"""
		color_intrinsics = self.intrinsic['RGB']
		depth_frame = self.frames['depth']
		color_image = self.frames['color_image']
		found_corners, points2D, color_to_send = cv_find_chessboard(color_image, self.chessboard_params)
		# cv2.imshow('chessborad_find',color_to_send)
		# cv2.waitKey(10)
		corners3D = [found_corners, None, None, None]
		try:
			if found_corners:
				points3D = np.zeros((3, len(points2D[0])))
				validPoints = [False] * len(points2D[0])
				for index in range(len(points2D[0])):
					corner = points2D[:,index].flatten()
					depth = get_depth_at_pixel(depth_frame, corner[0], corner[1]) # this is possible because we are using an aligned depth map to the RGB
					if depth != 0 and depth is not None:
						validPoints[index] = True
						[X,Y,Z] = convert_depth_pixel_to_metric_coordinate(depth, corner[0], corner[1], color_intrinsics)
						points3D[0, index] = X
						points3D[1, index] = Y
						points3D[2, index] = Z
				corners3D = [found_corners, points2D, points3D, validPoints]
		except Exception as e:
			print(e)
		return corners3D, color_to_send


	def perform_pose_estimation(self):
		"""
		Calculates the extrinsic calibration from the coordinate space of the camera to the 
		coordinate space spanned by a chessboard by retrieving the 3d coordinates of the 
		chessboard with the depth information and subsequently using the kabsch algortihm 
		for finding the optimal rigid transformation between the two coordinate spaces

		Returns:
		-----------
		retval : dict
		keys: str
			Serial number of the device
		values: [success, transformation, points2D, rmsd]
			success: bool
			transformation: Transformation
				Rigid transformation from the coordinate system of the camera to 
				the coordinate system of the chessboard
			points2D: array
				[2,N] array of the chessboard corners used for pose_estimation
			rmsd:
				Root mean square deviation between the observed chessboard corners and 
				the corners in the local coordinate system after transformation
		"""
		corners3D, color_to_send = self.get_chessboard_corners_in3d()
		corners3D = self.transform_3Dcamera_to_3Ddeepth(corners3D)
		found_corners, points2D, points3D, validPoints =  corners3D
		objectpoints = get_chessboard_points_3D(self.chessboard_params)
		retval = [False, None, None, None]
		if found_corners == True:
			#initial vectors are just for correct dimension
			valid_object_points = objectpoints[:,validPoints]
			valid_observed_object_points = points3D[:,validPoints]
			
			#check for sufficient points
			if valid_object_points.shape[1] < 5:
				print("Not enough points have a valid depth for calculating the transformation")
				
			else:
				[rotation_matrix, translation_vector, rmsd_value] = calculate_transformation_kabsch(valid_object_points, valid_observed_object_points)
				retval =[True, Transformation(rotation_matrix, translation_vector), points2D, rmsd_value]
				print("RMS error for calibration is :", rmsd_value, "m")

				if rmsd_value > 0.03:
					print("RMS error for calibration is too high: retry")
					retval[0] = False

		return retval, corners3D, color_to_send

	def transform_3Dcamera_to_3Ddeepth(self,points):
	
		found_corners, points2D, points3D, validPoints = points
		if found_corners:
			inverse = self.deep_to_rgb_extrinsic.inverse()
			points3D = inverse.apply_transformation(points3D)
		new_points = [found_corners,points2D,points3D,validPoints]
		return (new_points)