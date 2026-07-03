from setuptools import find_packages, setup

package_name = 'phd'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test', 'phd.backup', 'phd.backup.*']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ping2',
    maintainer_email='lcp123441@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'phd_ui = phd.phd_ui:main',
            'check_ai_dfm_dataset = phd.script.check_ai_direct_finger_motion_dataset:main',
            'train_ai_dfm_model = phd.script.train_ai_direct_finger_motion:main',
        ],
    },
)
