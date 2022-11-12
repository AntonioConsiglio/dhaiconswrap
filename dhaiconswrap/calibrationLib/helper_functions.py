##################################################################################################
##       License: Apache 2.0. See LICENSE file in root directory.		                      ####
##################################################################################################
##                  Box Dimensioner with multiple cameras: Helper files 					  ####
##################################################################################################

# Opencv helper functions and class
import cv2
import numpy as np

import warnings
warnings.filterwarnings("error")

"""
  _   _        _                      _____                     _    _
 | | | |  ___ | | _ __    ___  _ __  |  ___|_   _  _ __    ___ | |_ (_)  ___   _ __   ___
 | |_| | / _ \| || '_ \  / _ \| '__| | |_  | | | || '_ \  / __|| __|| | / _ \ | '_ \ / __|
 |  _  ||  __/| || |_) ||  __/| |    |  _| | |_| || | | || (__ | |_ | || (_) || | | |\__ \
 |_| |_| \___||_|| .__/  \___||_|    |_|    \__,_||_| |_| \___| \__||_| \___/ |_| |_||___/
				 _|
"""

def calculate_rmsd(points1, points2, validPoints=None):
	"""
	calculates the root mean square deviation between to point sets

	Parameters:
	-------
	points1, points2: numpy matrix (K, N)
	where K is the dimension of the points and N is the number of points

	validPoints: bool sequence of valid points in the point set.
	If it is left out, all points are considered valid
	"""
	assert(points1.shape == points2.shape)
	N = points1.shape[1]

	if validPoints == None:
		validPoints = [True]*N

	assert(len(validPoints) == N)

	points1 = points1[:,validPoints]
	points2 = points2[:,validPoints]

	N = points1.shape[1]

	dist = points1 - points2
	rmsd = 0
	for col in range(N):
		rmsd += np.matmul(dist[:,col].transpose(), dist[:,col]).flatten()[0]

	return np.sqrt(rmsd/N)


def get_chessboard_points_3D(chessboard_params):
	"""
	Returns the 3d coordinates of the chessboard corners
	in the coordinate system of the chessboard itself.

	Returns
	-------
	objp : array
		(3, N) matrix with N being the number of corners
	"""
	assert(len(chessboard_params) == 3)
	width = chessboard_params[0]
	height = chessboard_params[1]
	square_size = chessboard_params[2]
	objp = np.zeros((width * height, 3), np.float32)
	objp[:,:2] = np.mgrid[0:width,0:height].T.reshape(-1,2)
	return objp.transpose() * square_size/1000


def cv_find_chessboard(color_frame, chessboard_params):
	"""
	Searches the chessboard corners using the set infrared image and the
	checkerboard size

	Returns:
	-----------
	chessboard_found : bool
						  Indicates wheather the operation was successful
	corners          : array
						  (2,N) matrix with the image coordinates of the chessboard corners
	"""
	assert(len(chessboard_params) == 3)
	color_to_gray = cv2.cvtColor(color_frame,cv2.COLOR_RGB2GRAY)
	color_copy = color_frame.copy()
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
	chessboard_found = False
	chessboard_found, corners = cv2.findChessboardCorners(color_to_gray, (chessboard_params[0], chessboard_params[1]))

	if chessboard_found:
		corners = cv2.cornerSubPix(color_to_gray, corners, (5,5),(-1,-1), criteria)
		corners = np.transpose(corners, (2,0,1))
		if corners is not None:
			first = True
			cornersBis = corners.squeeze()
			cornersBis = cornersBis.T
			for corner in cornersBis:
				if not first:
					cv2.circle(color_copy,corner.astype(int),2,(0,255,0),-1)
				else:
					cv2.circle(color_copy,corner.astype(int),2,(0,0,255),-1)
					first = False

	return chessboard_found, corners, color_copy



def get_depth_at_pixel(depth_frame, pixel_x, pixel_y):
	"""
	Get the depth value at the desired image point

	Parameters:
	-----------
	depth_frame 	 : rs.frame()
						   The depth frame containing the depth information of the image coordinate
	pixel_x 	  	 	 : double
						   The x value of the image coordinate
	pixel_y 	  	 	 : double
							The y value of the image coordinate

	Return:
	----------
	depth value at the desired pixel

	"""

	return depth_frame[round(pixel_y), round(pixel_x)]


