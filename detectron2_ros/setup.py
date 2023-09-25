from glob import glob
import os

from setuptools import setup


package_name = 'detectron2_ros'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob('launch/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='juchajam',
    maintainer_email='juchajam@gmail.com',
    description="The detectron2_ros package.",
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
                'detectron2_ros = detectron2_ros.detectron2_ros:main',
        ],
    },
)

