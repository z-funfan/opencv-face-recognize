#!/usr/bin/env python
from setuptools import setup

setup(
    name='opencv-face-recognize',
    version='0.0.00.01',
    description='A demo program use openCV to recognize faces from camera',
    url='',
    author='Fengfan Zheng',
    author_email='fengfan_zheng@aisino-wincor.com',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.7',
    ],
    keywords='openCV face-recognize python',
    packages=['cv2', 'flask', 'face_recognition'],
)
