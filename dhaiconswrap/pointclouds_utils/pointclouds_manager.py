from .pointclouds_function_utils import *

class PointsCloudManager:
	def __init__(self,id):

		self.parameters = None
		self.sende = None
		self.receiver = None
		self.stopq = None
		self.viewROI = None
		self.id = id
		self._HasData = False
		self.pars = None
		self.zdirection = None
		self.options = None

	def SetParameters(self, calibration_info,roi_2D,viewROI,zdirection):
		self.calibration_info = calibration_info
		self.zdirection = zdirection
		self.pars = CalculatePointsCloudParameters(calibration_info, roi_2D, viewROI)


	def StartCalculation(self,frames):

		if self.pars is not None:

			result = CalculatePointsCloud(frames["depth"],
										  frames["color_image"],
										  self.pars,
										  False,
										  self.zdirection,
										  self.options,
										  Kdecimation=1,
										  ZmmConversion=1,
										  viewROI=self.viewROI,
										  )
			return result

	def set_options(self,depth_valid: float,z_threshold:float):
		"""
		- depth_valid: by default expressed in meters, define the max value of depth to consider.
		- z_threshold: by default expressed in meters, define the min value of z cordinate to consider.
		"""
		self.options = [depth_valid,z_threshold]

