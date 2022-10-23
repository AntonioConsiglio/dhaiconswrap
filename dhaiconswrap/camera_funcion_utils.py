import depthai as dhai
try:
	from .pointclouds_utils.pointclouds_manager import PointsCloudManager
	from .calibrationLib.calibration_function import load_calibration_json,check_calibration_exist
except:
	print('PointsCloudManager not loaded within calibration functions')
############################ CUSTOM EXCEPTIONS ###########################################

class BlobException(Exception):
	def __init__(self,message):
		super(BlobException,self).__init__(message)

############################ HELP FUNCTIONS ###########################################

def get_available_devices():

	available_devices = dhai.Device.getAllAvailableDevices()
	cameras_id = [device.getMxId() for device in available_devices]
	print(f'[CAMERAS FOUNDED] : {cameras_id}')
	return cameras_id

	############################ POINTCLOUD MANAGER FUNCTIONS ############################

def create_pointcloud_manager(id=None,calibrationInfo=None):

	if id is None:
		pointcloud_manager = PointsCloudManager('None')
	else:
		pointcloud_manager = PointsCloudManager('None')
	if not check_calibration_exist(idname=id):
		return None
	calibration_info,roi_2D,viewROI = load_calibration_json(id=id)
	pointcloud_manager.viewROI = viewROI
	calibrationInfo.append(calibration_info)
	pointcloud_manager.SetParameters(calibrationInfo,roi_2D,viewROI)

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

