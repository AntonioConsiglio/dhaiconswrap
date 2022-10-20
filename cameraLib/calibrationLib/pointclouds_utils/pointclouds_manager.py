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

	def SetParameters(self, calibration_info,roi_2D,viewROI):
		self.pars = CalculatePointsCloudParameters(calibration_info, roi_2D, viewROI)


	def PointsCloudManagerStartCalculation(self,depth_image,color_image,APPLY_ROI,
											Kdecimation,ZmmConversion,
											depth_threshold,viewROI):

		if self.pars is not None:

			result = CalculatePointsCloud(depth_image,
										  color_image,
										  self.pars,
										  [],
										  self.id,
										  APPLY_ROI,
										  Kdecimation,
										  ZmmConversion,
										  depth_threshold,
										  viewROI)
			if result:
				self._hasresult(result)

	def _hasresult(self,results):
		self.data = results
		self._HasData = True


	def HasData(self):
		return self._HasData

	def PointsCloudManagerGetResult(self):
		if self._HasData:
			self._HasData = False
			return self.data
		else:
			return None

