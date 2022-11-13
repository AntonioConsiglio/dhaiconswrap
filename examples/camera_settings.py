##########################################
##                                      ##
##    VERSION >= dhaiconswrap-0.2.5     ##
##                                      ##
##########################################
from dhaiconswrap import DeviceManager
import cv2
import os 

def main_easy_settings():
    device = DeviceManager(size = (640,480),
                           fps= 30,
                           deviceid=None,
                           calibration_mode=False,
                           verbose=True,
                           config_path=os.path.join(os.getcwd(),"examples","settings_files"))
    device.enable_device()
    ## POSSIBLE DEVICE SETTINGS:
    device.set_rgb_exposure(exposure=1500,iso=100) #Manual exposure for RGB sensor
    device.set_rgb_exposure(region=[200,200,300,300]) # Set the region for AutoExposure
    device.set_monos_exposure(exposure=1500,iso=100) #Manual exposure for mono sensors
    device.set_monos_exposure(region=[200,200,300,300]) # Set the region for AutoExposure
    device.set_focus(focus_value=150) # Set manual focus value
    device.set_conversion_depth(conv_factor=1) # set the output of Depth ==> Points cloud 
    ## POSSIBLE POINTS CLOUD SETTINGS:
    device.pointcloud_manager.set_options(depth_valid=3,z_threshold=2)

    for _ in range(50):
        res,frames,_ = device.pull_for_frames()
        if res:
            color = frames['color_image']
            disparity = frames['disparity_image']
            monoleft = frames["monos_image"]["left"]
            monoright = frames["monos_image"]["right"]
            cv2.imshow('color',color)
            cv2.imshow('disparity',disparity)
            cv2.imshow('monoleft',monoleft)
            cv2.imshow('monoright',monoright)
            cv2.waitKey(1)
    cv2.destroyAllWindows()
    device.disable_device()    

if __name__ == '__main__':
    main_easy_settings()