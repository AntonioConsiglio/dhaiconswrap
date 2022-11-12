##########################################
##                                      ##
##    VERSION => dhaiconswrap-0.2.5     ##
##                                      ##
##########################################

from dhaiconswrap import DeviceManager
import cv2
import os
import numpy as np

## USED ONLY SSD MODELS - FOR YOLO MODEL LOOK THE OTHER EXAMPLE

BLOB_PATH = os.path.join("examples","blob_path","mobilenet-ssd_openvino_2021.2_6shave.blob")

def main_neural_network():
    device = DeviceManager(size = (640,480),
                           fps= 30,
                           nn_mode=True,
                           blob_path=BLOB_PATH)
    device.enable_device()
    # set the names of your class labeled
    device.set_labels_names([i*"person" for i in np.ones(20,dtype=int)])
    # SET DEPTH VALID AND Z THRESHOLD
    device.pointcloud_manager.set_options(3,0.5)

    for _ in range(50):
        # if the get_pointscloud flag is True, you will se the spatial position of your object when depth values allow to calculate it
        # if you want to output the image with bounding box and spatial location, the write_detections need to be set to True
        # Because all the computation is done with CPU, you should take it into account.
        res,frames,results = device.pull_for_frames(get_pointscloud=True,write_detections=True)
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