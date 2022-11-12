##########################################
##                                      ##
##    VERSION => dhaiconswrap-0.2.5     ##
##                                      ##
##########################################

from dhaiconswrap import DeviceManager
import cv2
import os

## USED ONLY SSD MODELS - FOR YOLO MODEL LOOK THE OTHER EXAMPLE

BLOB_PATH = os.path.join("blob_path","mobilenet-ssd_openvino_2021.2_6shave.blob")

def main_neural_network():
    device = DeviceManager(size = (640,480),
                           fps= 30,
                           nn_mode=True,
                           blob_path=BLOB_PATH)
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
    main_neural_network()