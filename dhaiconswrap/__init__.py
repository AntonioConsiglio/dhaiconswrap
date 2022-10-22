"""
dhaiconswrap.

A python wrapper for OAK-Luxonis cameras.
Calibration function and Points Cloud manager can be used with other cameras if you use the same method names.
"""
__version__ = "0.1.13"
__author__ = "Consiglio Antonio"

# try:
#     __import__('pkg_resources').declare_namespace(__name__)
# except ImportError:
#     __path__ = __import__('pkgutil').extend_path(__path__, __name__)
    
from .calibrationLib import docalibration
from .pointclouds_utils import PointsCloudManager
from .cameraManager import DeviceManager,get_available_device


