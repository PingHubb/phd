from setuptools import find_packages, setup

package_name = 'phd'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=[
        'test', 'archive', 'archive.*', 'phd.backup', 'phd.backup.*',
    ]),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ping2',
    maintainer_email='lcp123441@gmail.com',
    description='Tactile sensing, visualization, and robot-control research interface',
    license='Proprietary',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'phd_ui = phd.phd_ui:main',
            'pinglab_tools = phd.script.__main__:main',
            'check_ai_dfm_dataset = phd.script.analysis.check_ai_direct_finger_motion_dataset:main',
            'train_ai_dfm_model = phd.script.training.train_ai_direct_finger_motion:main',
            'train_tactile_proximity = phd.script.training.train_tactile_proximity:main',
            'train_touch_tracking = phd.script.training.train_touch_tracking:main',
            'setup_goodix_usb_permissions = phd.script.hardware.setup_goodix_usb_permissions:main',
            'analyse_height_profile = phd.script.analysis.analyse_height_profile:main',
            'analyse_proximity_profile = phd.script.analysis.analyse_proximity_profile:main',
        ],
    },
)
