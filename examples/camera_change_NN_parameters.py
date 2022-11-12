##########################################
##                                      ##
##    VERSION >= dhaiconswrap-0.2.5     ##
##                                      ##
##########################################
from dhaiconswrap import DeviceManager
import depthai as dhai
import cv2
import os

BLOB_PATH = os.path.join("blob_path","mobilenet-ssd_openvino_2021.2_6shave.blob")

'''
THE WAY YOU CAN CHANGE NEURAL NETWORK INITIALIZATION IS OVERRIDING THE CONFIGURE FUNCTION OF DEVICE MANAGER CLASS 
'''

class PersonalDeviceManager(DeviceManager):

	def __init__(self,**kargs):
		super(PersonalDeviceManager,self).__init__(**kargs)
	
	## THIS FUNCTION FOR CONFIGURE THE INPUT IMAGE SIZE AND CHANNELS

	def _configure_image_manipulator(self,pipeline,verbose):
		size = self.depthconfig["nn_size"]
		manip = pipeline.create(dhai.node.ImageManip)
		manipOut = pipeline.create(dhai.node.XLinkOut)
		manipOut.setStreamName('neural_input')
		manip.initialConfig.setResize(*size)
		manip.initialConfig.setFrameType(dhai.ImgFrame.Type.BGR888p)
		manip.out.link(manipOut.input)
		
		return manip,manipOut

	## THIS FUNCTION FOR CONFIGURE THE NEURAL NETWORK NODE

	def _configure_nn_node(self,manip,pipeline,blob_path,verbose):
			
			nn = pipeline.create(dhai.node.MobileNetDetectionNetwork)
			nnOut = pipeline.create(dhai.node.XLinkOut)
			nnOut.setStreamName("neural")
			# define nn features
			nn.setConfidenceThreshold(self.depthconfig["nn_threshold"])
			nn.setBlobPath(blob_path)
			nn.setNumInferenceThreads(2)
			# Linking
			manip.out.link(nn.input)
			nn.out.link(nnOut.input)

def main_easy():
	device = PersonalDeviceManager(size = (640,480),
									fps= 30,
									nn_mode = True,
									blob_path = BLOB_PATH,
									)
	device.enable_device()
	for _ in range(50):
		res,frames,results = device.pull_for_frames()
		if res:
			color = frames['color_image']
			disparity = frames['disparity_image']
			detections = results['detections']
			cv2.imshow('color',color)
			cv2.imshow('disparity',disparity)
			cv2.waitKey(1)
	cv2.destroyAllWindows()    

if __name__ == '__main__':
	main_easy()