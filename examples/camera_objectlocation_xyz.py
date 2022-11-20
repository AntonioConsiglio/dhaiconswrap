##########################################
##                                      ##
##    VERSION >= dhaiconswrap-0.2.5     ##
##                                      ##
##########################################
from dhaiconswrap import DeviceManager,get_available_devices,domulticalibration
from dhaiconswrap.camera_funcion_utils import visualise_Axes
import cv2
import os

CHESSBOARD_PARAMETERS = [16,9,15] #List of ColumnsCorners, RowCorners, SquareWidth of chessboard
CALIBRATION_ROI =  [0,0,1080,1920] #The RGB image for calibration is 1920x1080. 
SHIFT_CALIBRATION = [0,0,0] # Trasport vector, if is needed to set the axes center in another position instead of the chessboard corner
CALIBRATION_PATH = os.path.join(os.getcwd(),"examples","settings_files")# Where the calibration .json file are stored after the calibration procces. This path is used also for load it.
                        #Is adviced to leave the default folder or you need to override the class methods used to call these files.


## MULTIPLE CAM CALIBRATION
def multi_cam_calibration():
    devices: dict[str,DeviceManager] = {}
    devices_names = get_available_devices()

    for dev_name in devices_names:
        device = DeviceManager(size = (1920,1080),
                            fps= 30,
                            deviceid=dev_name,
                            calibration_mode=True,
                            verbose=True,
                            config_path=os.path.join(os.getcwd(),"examples","settings_files"))
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

## MULTIPLE CAM SETUP AND RUNTIME

def multi_easy():
    devices:dict[str,DeviceManager] = {}
    deviceIDs = get_available_devices()

    for devID in deviceIDs:
        device = DeviceManager(size = (640,480),
                            fps= 30,
                            deviceid=devID,
                            calibration_mode=False,
                            verbose=True,
                            config_path=os.path.join(os.getcwd(),"examples","settings_files"))
        device.enable_device(usb2Mode=False)
        devices[devID] = device
    # SET DEPTH VALID AND Z THRESHOLD
    device.pointcloud_manager.set_options(5,-1.0)

    #RUNTIME
    for _ in range(150):
        for devId,device in devices.items():
            res,frames,results = device.pull_for_frames()
            if res:
                color = frames['color_image']
                # LOCATION FOR DETECTIONS
                cordinates = device.determinate_object_location(image_to_write=color,
                                                    points_cloud_data=results["points_cloud_data"],
                                                    detections=[[0,0.99,[450,400,600,478]]])
                disparity = frames['disparity_image']
                cv2.imshow(f'color - {devId}',visualise_Axes(color,device.pointcloud_manager.calibration_info))
                cv2.imshow(f'disparity - {devId}',disparity)
                cv2.waitKey(1)
    cv2.destroyAllWindows()
    device.disable_device()       

if __name__ == '__main__':

    #_ = multi_cam_calibration()
    multi_easy()