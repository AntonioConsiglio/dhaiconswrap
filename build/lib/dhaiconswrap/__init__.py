"""
dhaiconswrap.

A python wrapper for OAK-Luxonis cameras.
"""

__version__ = "0.1.8"
__author__ = 'Antonio Consiglio'


__import__('calibrationLib').declare_namespace(__name__)
__import__('pointclouds_utils').declare_namespace(__name__)
__import__('cameraManager').declare_namespace(__name__)
