from .pointclouds_function_utils import *

#import ptvsd

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


	def PointsCloudManagerStartCalculation(self,depth_image,color_image,APPLY_ROI,
											Kdecimation,ZmmConversion,viewROI):

		if self.pars is not None:

			result = CalculatePointsCloud(depth_image,
										  color_image,
										  self.pars,
										  APPLY_ROI,
										  self.zdirection,
										  self.options,
										  Kdecimation,
										  ZmmConversion,
										  viewROI,
										  )
			if result:
				self._hasresult(result)
			
			return result

	def _hasresult(self,results):
		self.data = results
		self._HasData = True


	def HasData(self):
		return self._HasData

	def set_options(self,depth_valid: float,z_threshold:float):
		"""
		- depth_valid: by default expressed in meters, define the max value of depth to consider.
		- z_threshold: by default expressed in meters, define the min value of z cordinate to consider.
		"""
		self.options = [depth_valid,z_threshold]

	def PointsCloudManagerGetResult(self):
		if self._HasData:
			self._HasData = False
			return self.data
		else:
			return None

