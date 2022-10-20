from setuptools import setup

setup(
    name='dhaiconswrap',
    version='0.1.0',    
    description='A depthai wrapper with calibration and points cloud functions',
    url='https://github.com/AntonioConsiglio/dhaiconswrap',
    author='Antonio Consiglio',
    author_email='consiglio.antonio.bi@gmail.com',
    license='LGPL',
    packages=['dhaiconswrap'],
    install_requires=['depthai>=2.15.0.0',
                      'numpy>=1.22.1'
                      'opencv-python>=4.5.5.64'
                      'opencv-contrib-python>=4.5.5.64',                     
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: LGPL License',  
        'Operating System :: POSIX :: Linux :: Windows',        
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)