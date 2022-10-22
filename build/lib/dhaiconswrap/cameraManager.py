try:
	import depthai as dhai
except:
	print('depthai library is not installed!!')
	print('Install depthai library ... "pip install depthai"')
import cv2
import numpy as np
import os

try:
	from .calibrationLib.calibration_kabsch import Transformation
except:
	print('Impossible to import Transformation class from ..calibrationLib.calibration_kabsch ')

from .camera_funcion_utils import configure_rgb_sensor,configure_depth_sensor,create_pointcloud_manager


# Help functions

def get_available_device():

	available_devices = dhai.Device.getAllAvailableDevices()
	cameras_id = [device.getMxId() for device in available_devices]
	print(cameras_id)

class IntrinsicParameters():

	def __init__(self,intrinsic_info,size):

		self.fx = intrinsic_info[0][0]
		self.cx = intrinsic_info[0][2]
		self.fy = intrinsic_info[1][1]
		self.cy = intrinsic_info[1][2]
		self.h = size[1]
		self.w = size[0]

class DeviceManager():

	def __init__(self,size,fps,nn_mode=False,calibration_mode = False,blob_path=None):
		self.pipeline = dhai.Pipeline()
		self.size = size
		self.fps = fps
		self.nn_active = nn_mode
		self.calibration = calibration_mode
		self.zmmconversion = 1000
		self.BLOB_PATH = blob_path
		self._configure_device()
		self.node_list = self.pipeline.getNodeMap()

	def _configure_device(self):

		self._configure_rgb_sensor()
		self._configure_depth_sensor()
		
	
	def _configure_rgb_sensor(self):
		if self.nn_active:
			configure_rgb_sensor(self.pipeline,self.size,self.fps,self.nn_active,self.BLOB_PATH,self.calibration)
		else:
			configure_rgb_sensor(self.pipeline,self.size,self.fps,self.nn_active,self.calibration)
	
	def _configure_depth_sensor(self):
		configure_depth_sensor(self.pipeline,self.calibration)

	def enable_device(self):
		self.device_ = dhai.Device(self.pipeline,usb2Mode=True)
		if self.nn_active:
			self.max_disparity = self.node_list[8].initialConfig.getMaxDisparity()
		else:
			self.max_disparity = self.node_list[4].initialConfig.getMaxDisparity()
		self._set_output_queue()
		self.get_intrinsic()
		self.get_extrinsic()
		calibration_info = [self.intrinsic_info['RGB'],self.intrinsic_info['RIGHT'],self.extrinsic_info]
		self.pointcloud_manager = create_pointcloud_manager('first_camera',calibration_info)

	def _set_output_queue(self):
		self.q_rgb = self.device_.getOutputQueue("rgb",maxSize = 1,blocking = False)
		self.q_depth = self.device_.getOutputQueue("depth",maxSize = 1,blocking = False)
		self.q_disparity = self.device_.getOutputQueue("disparity",maxSize=1,blocking=False)
		if self.nn_active:
			self.q_nn = self.device_.getOutputQueue('neural',maxSize=1,blocking=False)
			self.q_nn_input = self.device_.getOutputQueue('neural_input',maxSize=1,blocking=False)

	def _normalize_detections(self,detections):
		det_normal = []
		for detection in detections:
			label = detection.label
			score = detection.confidence
			xmin,ymin,xmax,ymax = detection.xmin,detection.ymin,detection.xmax,detection.ymax
			xmin,xmax = int(xmin*self.size[0]),int(xmax*self.size[0])
			ymin,ymax = int(ymin*self.size[1]),int(ymax*self.size[1])
			det_normal.append([label,score,[xmin,ymin,xmax,ymax]])

		return det_normal

	def _write_detections_on_image(self,image,detections):
		for detection in detections:
			xmin,ymin,xmax,ymax = detection[2]
			label = detection[0]
			score = detection[1]
			cv2.rectangle(image,(xmin,ymin),(xmax,ymax),(255,255,255),2)
			cv2.putText(image,f'{label}: {round(score*100,2)} %',(xmin,ymin-10),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)

	def set_conversion_depth(self,conv_factor):
		'''
			default value is 1000 which means the output will be in mm as default
		'''
		self.zmmconversion = conv_factor

	def _convert_depth(self,depth):
		depth = depth.flatten()/self.zmmconversion
		depth = np.reshape(depth,(self.size[1],self.size[0]))
		return depth

	def determinate_object_location(self,image_to_write,points_cloud_data,detections,offset=10):
		'''
			input:\n
			image_to_write: image where the average position is written
			points_cloud_dat: the points cloud results, stored inside a dictionary as in results \n
							obtained using the pull for frames method\n
			detections: list of detections\n
			offset: default = 10, the offset from the center of the detection to take the points cloud value\n
					and averaging them to output the position in the space of the object\n
			Output:\n
			cordinates: list of finded cordinates of the obects 		
		'''
		cordinates = []
		xyz_points = points_cloud_data['XYZ_map_valid']
		for detection in detections:
			xmin,ymin,xmax,ymax = detection[2]
			xcenter = (xmin+((xmax-xmin)//2))
			ycenter = (ymin+((ymax-ymin)//2))
			useful_value = xyz_points[ycenter-offset:ycenter+offset,xcenter-offset:xcenter+offset]
			useful_value = useful_value.reshape((useful_value.shape[0]*useful_value.shape[1],3))
			useful_value = useful_value[np.any(useful_value != 0,axis=1)]
			if useful_value.size == 0:
				continue 
			elif useful_value.shape[0] >1:
				avg_pos_obj = np.mean(useful_value,axis=0)*1000
			else:
				avg_pos_obj= useful_value[0]*1000
			avg_pos_obj = avg_pos_obj.astype(int)
			if not np.all(avg_pos_obj == 0):
				cordinates.append(avg_pos_obj.tolist())
				try:
					x,y,z = avg_pos_obj.tolist()
					cv2.putText(image_to_write,f"x: {x} mm",(xcenter+8,ycenter-30),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),2)
					cv2.putText(image_to_write,f'y: {y} mm',(xcenter+8,ycenter-15),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),2)
					cv2.putText(image_to_write,f'z: {z} mm',(xcenter+8,ycenter),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),2)
				except Exception as e:
					print(f"[CALCULATE OBJECT LOCATION]: {e}")
			
		return cordinates

	def pull_for_frames(self):
		'''
		- output:\n
			frame_state: bool \n
			frames: dict[color_image,depth,disparity_image]
			results: dict['points_cloud_data','detections']
		'''
		frames = {}
		state_frame = False
		frame_count = 0

		while not state_frame:
			rgb_foto = self.q_rgb.tryGet()
			depth = self.q_depth.get()
			disparity_frame = self.q_disparity.get()
			nn_foto = None
			if self.nn_active:
				nn_foto = self.q_nn_input.tryGet()
				nn_detection = self.q_nn.tryGet()
				if nn_detection is not None:
					detections = nn_detection.detections
				else:
					detections = None

			if rgb_foto is not None and depth is not None:

				state_frame = True
				frames['color_image'] = rgb_foto.getCvFrame()
				frames['depth'] = self._convert_depth(depth.getFrame())
				frames['disparity_image'] = disparity_frame.getFrame()#*(255 /self.max_disparity)).astype(np.uint8)
				results = {}
				results['points_cloud_data'] = None
				results['detections'] = None
				if self.pointcloud_manager is not None:
					results['points_cloud_data'] = self.pointcloud_manager.PointsCloudManagerStartCalculation(depth_image=frames['depth'],
														color_image=frames['color_image'],
														APPLY_ROI=False,
														Kdecimation=1,
														ZmmConversion=1,
														depth_threshold=0.001,
														viewROI=self.pointcloud_manager.viewROI
														)
				if nn_foto is not None:
					frames['nn_input'] = nn_foto.getCvFrame()
					if detections is not None:
						results['detections'] = self._normalize_detections(detections)
						self._write_detections_on_image(frames['color_image'],results['detections'])
						
				return state_frame,frames,results
			else:
				frame_count += 1
				if frame_count > 10:
					print('empty_frame: ',frame_count)
				return False,None,None

	def get_intrinsic(self):
		self.intrinsic_info = {}
		calibration_info = self.device_.readCalibration()
		intr_info_rgb = calibration_info.getCameraIntrinsics(dhai.CameraBoardSocket.RGB,resizeHeight=self.size[1],resizeWidth=self.size[0])
		intr_info_right = calibration_info.getCameraIntrinsics(dhai.CameraBoardSocket.RIGHT,resizeHeight=self.size[1],resizeWidth=self.size[0])
		self.intrinsic_info['RGB'] = IntrinsicParameters(intr_info_rgb,self.size)
		self.intrinsic_info['RIGHT'] = IntrinsicParameters(intr_info_right,self.size)

	def get_extrinsic(self):
		calibration_info = self.device_.readCalibration()
		extrin_info = np.array(calibration_info.getCameraExtrinsics(dhai.CameraBoardSocket.RIGHT,dhai.CameraBoardSocket.RGB))
		extrin_info[:,3] = extrin_info[:,3]/1000 
		self.extrinsic_info = Transformation(trasformation_mat=extrin_info)

	def get_intrisic_and_extrinsic(self):
		return self.intrinsic_info,self.extrinsic_info

