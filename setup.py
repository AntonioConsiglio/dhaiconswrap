from setuptools import setup,find_packages

setup(
    name='dhaiconswrap',
    version='0.1.11',    
    description='A depthai wrapper with calibration and points cloud functions',
    url='https://github.com/AntonioConsiglio/dhaiconswrap',
    author='Antonio Consiglio',
    author_email='consiglio.antonio.bi@gmail.com',
    license='MIT',
    # namespace_packages=['dhaiconswrap'],
    packages= find_packages(),
    install_requires=['depthai>=2.15.0.0',
                      'numpy>=1.22.1',
                      'opencv-python>=4.5.5.64',
                      'opencv-contrib-python>=4.5.5.64',                     
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',  
        'Operating System :: POSIX :: Linux', 
        'Operating System :: Microsoft :: Windows :: Windows 10',       
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)