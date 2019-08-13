from setuptools import setup

setup(
    name='rl_algorithms',
    py_modules=['rl_algorithms'],
    version='0.1',
    install_requires=[
        'gym[atari,box2d,classic_control]',
        'matplotlib',
        'mpi4py',
        'pandas',
        'numpy',
        'roboschool',
        'box2d-py',
        'tensorflow>=1.13,<2.0',
        'opencv-python'
    ],
    description="A collection of RL algorithms",
    author="Ollie Day",
)