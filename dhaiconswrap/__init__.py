"""
dhaiconswrap.

A python wrapper for OAK-Luxonis cameras.
Calibration function and Points Cloud manager can be used with other cameras if you use the same method names.
"""
__version__ = "0.2.5"
__author__ = "Consiglio Antonio"

# try:
#     __import__('pkg_resources').declare_namespace(__name__)
# except ImportError:
#     __path__ = __import__('pkgutil').extend_path(__path__, __name__)
    
from .calibrationLib import docalibration,domulticalibration
from .pointclouds_utils import PointsCloudManager
from .cameraManager import DeviceManager
from .camera_funcion_utils import get_available_devices


