##########################################
##                                      ##
##    VERSION >= dhaiconswrap-0.2.5     ##
##                                      ##
##########################################
from dhaiconswrap import DeviceManager,docalibration,get_available_devices,domulticalibration

CHESSBOARD_PARAMETERS = [16,9,15] #List of ColumnsCorners, RowCorners, SquareWidth of chessboard
CALIBRATION_ROI =  [0,0,1080,1920] #The RGB image for calibration is 1920x1080. 
SHIFT_CALIBRATION = [0,0,0] # Trasport vector, if is needed to set the axes center in another position instead of the chessboard corner
CALIBRATION_PATH = "./" # Where the calibration .json file are stored after the calibration procces. This path is used also for load it.
                        #Is adviced to leave the default folder or you need to override the class methods used to call these files.

## SINGLE CAM CALIBRATION
def main_calibration():
    device = DeviceManager(size = (1920,1080),
                           fps= 30,
                           calibration_mode=True,
                           verbose=True)
    device.enable_device()

    _ = docalibration(device_manager=device,
                      chessboard_params=CHESSBOARD_PARAMETERS,
                      calibration_roi=CALIBRATION_ROI,
                      shiftcalibration=SHIFT_CALIBRATION,
                      zdirection=False,
                      path=CALIBRATION_PATH)
    device.disable_device()

## MULTIPLE CAM CALIBRATION

def multi_cam_calibration():
    devices: dict[str,DeviceManager] = {}
    devices_names = get_available_devices()

    for dev_name in devices_names:
        device = DeviceManager(size = (1920,1080),
                            fps= 30,
                            deviceid=dev_name,
                            calibration_mode=True,
                            verbose=True)
        device.enable_device()
        devices[dev_name] = device
    
    _ = domulticalibration(device_managers=devices,
                            chessboard_params=CHESSBOARD_PARAMETERS,
                            calibration_roi=CALIBRATION_ROI,
                            shiftcalibration=SHIFT_CALIBRATION,
                            zdirection=True,
                            path=CALIBRATION_PATH) 

    for _,device in devices.items():
        device.disable_device()       

if __name__ == '__main__':
    _ = main_calibration()
    _ = multi_cam_calibration()