import numpy as np
import time
#import ptvsd

from ..calibrationLib.calibration_kabsch import Transformation

def CalculatePointsCloudParameters(calibration_info, roi_2D, viewROIcoords):
	#ptvsd.debug_this_thread()
	depth_intrinsics = calibration_info[1]
	color_intrinsics = calibration_info[0]
	depth2color_extrinsics_mat = calibration_info[2]
	camera2world_mat = calibration_info[3]
	w = depth_intrinsics.w
	h = depth_intrinsics.h

	depth2color_trans = depth2color_extrinsics_mat# Transformation(trasformation_mat = depth2color_extrinsics_mat)
	#camera2world_mat = Transformation(trasformation_mat = camera2world_mat)

	nx = np.linspace(0, w-1, w)
	ny = np.linspace(0, h-1, h)
	u, v = np.meshgrid(nx, ny)
	u_flat = u.flatten()
	v_flat = v.flatten()

	x_norm = (u_flat - depth_intrinsics.cx)/depth_intrinsics.fx
	y_norm = (v_flat - depth_intrinsics.cy)/depth_intrinsics.fy

	ROI_valid = np.zeros((h, w), dtype=bool)
	ROI_valid[viewROIcoords[1]:viewROIcoords[3],viewROIcoords[0]:viewROIcoords[2]] = True
	ROI_valid = ROI_valid.flatten()

	return color_intrinsics, depth2color_trans, camera2world_mat, roi_2D, x_norm, y_norm, w, h, ROI_valid

def CalculatePointsCloud(depth_image, color_image, pars, addinfo,device,APPLY_ROI,
							Kdecimation,ZmmConversion,depth_threshold,viewROI):
	'''
		pars = parameters based on CalculatePointsCloudParameters
		addinfo = dizionario proveniente dalla captazione di eventi da tastiera \n durante il running del codice per visualizzare o no qualcosa.
	'''
	#ptvsd.debug_this_thread()
	color_intrinsics, depth2color_trans, cam2world_mat, roi_2D, x_norm, y_norm, w, h, viewROI_map = pars
	if APPLY_ROI:
		viewROI_map_color = viewROI_map.reshape((h,w))
		newH = np.max(np.sum(viewROI_map_color,axis=0))
		newW= np.max(np.sum(viewROI_map_color,axis=1))
		inittime = time.time()


	# create empty arrays
	RGB_map_valid = np.zeros((h, w, 3), dtype=np.uint8)
	XYZ_map_valid = np.zeros((h, w, 3), dtype=np.float32)

	# take all z coords
	z = depth_image.flatten()

	# apply a filter of valid point detected (0 value mean no distance found)
	if APPLY_ROI:
		filter_valid = np.nonzero(np.logical_and(z > 0, z < 5000) & viewROI_map)[0]
	else:
		#filter_valid = np.nonzero(np.logical_and(z > 0, z < 5000))[0]
		filter_valid = np.nonzero(np.logical_and(z > 0, z < 0.650))[0]


	if Kdecimation > 1:
		size = (len(filter_valid) // Kdecimation) * Kdecimation # Rendo la size divisibile per 4 perfettamente 
		filter_valid = filter_valid[:size].reshape(-1,Kdecimation).max(1) 

	# filter by valid points and calculate x,y,z coords
	z = z[filter_valid]/ZmmConversion # 1000 conversione in metri
	# Ricavo la x che avevamo normalizzato in precedenza con la z
	x = np.multiply(x_norm[filter_valid],z)
	# Ricavo la y che era stata normalizzata in precedenza con la z (o meglio era stata normalizzata secondo la focal lenth)
	y = np.multiply(y_norm[filter_valid],z)
	# compose the xyz array
	point_cloud_xyz = np.asanyarray((x,y,z))

	
	#convert point from camera coords to world coords
	XYZ_world_cloud_valid = cam2world_mat.apply_transformation(point_cloud_xyz)

	# calculate points index based on z. The z coord direction in pointing to lower, so coords shall be negative
	filter0 = XYZ_world_cloud_valid[2, :] <-depth_threshold
	XYZ_world_cloud_valid = XYZ_world_cloud_valid[:,filter0]
	XYZ_cloud_valid = point_cloud_xyz[:,filter0]

	if APPLY_ROI:
		# filter by x coords of ROI
		filter1 = np.logical_and(XYZ_world_cloud_valid[0,:]<roi_2D[1], XYZ_world_cloud_valid[0,:]>roi_2D[0])
		XYZ_world_cloud_valid = XYZ_world_cloud_valid[:,filter1]
		XYZ_cloud_valid = XYZ_cloud_valid[:,filter1]

		# filter by y coords of ROI
		filter2 = np.logical_and(XYZ_world_cloud_valid[1,:]<roi_2D[3], XYZ_world_cloud_valid[1,:]>roi_2D[2])
		XYZ_world_cloud_valid = XYZ_world_cloud_valid[:,filter2]
		XYZ_cloud_valid = XYZ_cloud_valid[:,filter2]

	# move to system of coords of RGB camera
	XYZ_cloud_valid_as_RGB = depth2color_trans.apply_transformation(XYZ_cloud_valid)

	# convert to pixel coords (centered around focal point of camera)
	z_RGB = XYZ_cloud_valid_as_RGB[2,:]
	x_RGB = np.divide(XYZ_cloud_valid_as_RGB[0,:],z_RGB)
	y_RGB = np.divide(XYZ_cloud_valid_as_RGB[1,:],z_RGB)

	# move to pixel coords (u, v)
	v = (x_RGB * color_intrinsics.fx + color_intrinsics.cx).astype(int)
	u = (y_RGB * color_intrinsics.fy + color_intrinsics.cy).astype(int)

	filter = (v >= 0) & (v <= w - 1)

	v = v[filter]
	u = u[filter]
	XYZ_world_cloud_valid = XYZ_world_cloud_valid[:,filter]

	filter = (u >= 0) & (u <= h - 1)

	v = v[filter]
	u = u[filter]
	XYZ_world_cloud_valid = XYZ_world_cloud_valid[:,filter]


	# get only the point valids
	RGB_cloud_valid = color_image[u, v]

	RGB_map_valid[u,v] = RGB_cloud_valid

	XYZ_map_valid[u,v] = XYZ_world_cloud_valid.T

	pointcloud_results = {}

	pointcloud_results['color_image'] = color_image
	if APPLY_ROI:
		pointcloud_results['color_image_video'] = viewROI
	else:
		pointcloud_results['color_image_video'] = viewROI
	pointcloud_results['depth_image'] = depth_image
	pointcloud_results['XYZ_world_cloud_valid'] = XYZ_world_cloud_valid
	pointcloud_results['RGB_cloud_valid'] = RGB_cloud_valid
	pointcloud_results['XYZ_map_valid'] = XYZ_map_valid
	pointcloud_results['RGB_map_valid'] = RGB_map_valid
	pointcloud_results['inittime'] = 0
	pointcloud_results['addinfo'] = addinfo
	pointcloud_results['dodetection'] = True

	return pointcloud_results


