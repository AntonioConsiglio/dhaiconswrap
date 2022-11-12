##########################################
##                                      ##
##    VERSION >= dhaiconswrap-0.2.5     ##
##                                      ##
##########################################
from dhaiconswrap import DeviceManager,get_available_devices
import cv2

def multi_easy():
    devices:dict[str,DeviceManager] = {}
    deviceIDs = get_available_devices()

    for devID in deviceIDs:
        device = DeviceManager(size = (640,480),
                               fps= 30,
                               deviceid=devID,
                               calibration_mode=False,
                               verbose=True)
        device.enable_device(usb2Mode=False)
        devices[devID] = device
    
    for _ in range(50):
        for devId,device in devices.items():
            res,frames,_ = device.pull_for_frames()
            if res:
                color = frames['color_image']
                disparity = frames['disparity_image']
                cv2.imshow(f'color - {devId}',color)
                cv2.imshow(f'disparity - {devId}',disparity)
                cv2.waitKey(1)
    cv2.destroyAllWindows()
    device.disable_device()    

if __name__ == '__main__':
    multi_easy()