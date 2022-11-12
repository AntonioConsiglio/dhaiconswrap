##########################################
##                                      ##
##    VERSION >= dhaiconswrap-0.2.5     ##
##                                      ##
##########################################
from dhaiconswrap import DeviceManager
import cv2

def main_easy():
    device = DeviceManager(size = (640,480),
                           fps= 30,
                           deviceid=None,
                           calibration_mode=False,
                           verbose=True)
    device.enable_device()
    for _ in range(50):
        res,frames,_ = device.pull_for_frames()
        if res:
            color = frames['color_image']
            disparity = frames['disparity_image']
            cv2.imshow('color',color)
            cv2.imshow('disparity',disparity)
            cv2.waitKey(1)
    cv2.destroyAllWindows()
    device.disable_device()    

if __name__ == '__main__':
    main_easy()