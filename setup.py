from setuptools import setup

setup(
    name='gym_collision_avoidance',
    version='1.0.0',
    description='Simulation environment for collision avoidance',
    url='https://github.com/mit-acl/gym-collision-avoidance',
    author='Michael Everett, Yu Fan Chen, Jonathan P. How, MIT',  # Optional
    keywords='robotics planning gym rl',  # Optional
    python_requires='>=3.0, <4',
    install_requires=[
        'tensorflow==1.15.2',
        'scipy==1.2.2',
        'requests',
        'Pillow',
        'PyOpenGL',
        'pyyaml',
        'matplotlib>=3.0.0',
        'shapely',
        'pytz',
        'imageio==2.4.1',
        'gym',
        'moviepy',
        'pandas',
        'networkx',
        'torch===1.3.1', #compatibility #used 1.2 instead of 1.3 #cuda compatibilty
        'attrdict',
    ],
)
