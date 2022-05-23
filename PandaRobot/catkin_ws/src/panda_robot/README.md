# Panda Robot [![Python 2.7, 3.6+](https://img.shields.io/badge/python-2.7,%203.6+-blue.svg?logo=python)](https://www.python.org/downloads/release/python-360/) [![ROS Version](https://img.shields.io/badge/ROS-Melodic,%20Noetic-brightgreen.svg?logo=ros)](https://ros.org/)


A Python interface package built over the [*Franka ROS Interface*] package, combining its different classes to provide a unified interface for controlling and handling the Franka Emika Panda robot. Also works directly with [*Panda Simulator*].

The package provides an extensive and unified [API] for controlling and managing the Franka Emika Robot (and gripper) using pre-defined low-level controllers (position, velocity, torque, joint impedance), MoveIt planners, and JointTrajectory action service.

*NOTE: This version requires [Franka ROS Interface v0.7.1] to be installed.*

## Features

- Provides simple-intuitive interface classes with methods to directly and easily control the robot using low-level controllers, MoveIt planners, or Trajectory action client.
- Get real-time robot state, joint state, controller state, kinematics, dynamics, etc.
- Provides Kinematics computation (using [KDL library](http://wiki.ros.org/kdl)). Automatically adjusts computations for the end-effector frames set in Dash or by code.
- Integrated with gripper control.
- Manage frames transformation and controller switching using simple utility functions.
- Works directly on simulated robot when using [*Panda Simulator*] providing direct sim-to-real and real-to-sim code transfer.

## Installation

**NOTE:** This branch should work with ROS Melodic and ROS Noetic. Tested on:

| ROS Version | Required Python Version |
|-------------|-------------------------|
| Melodic     | 2.7+                    |
| Noetic      | 3.6+                    |

**The following dependencies have to be met before installing PandaRobot**:

  - Requires ROS Melodic or Noetic (preferably the `desktop-full` version to cover all dependencies such as PyKDL and MoveIt)

  - [*Franka ROS Interface*] Installing this package correctly would also resolve all the other dependencies for PandaRobot.*

Once the dependencies are installed, the package can be installed either from pypi, or by building from source. Note that the installation may be successful even if the above dependencies are not met, but the package cannot be used until the dependencies are installed.


See script (`test/test_pos_controllers.py`) to see how the robot can be controlled using low-level joint controllers.

See script (`scripts/env.py`), and run it interactively (`python -i env.py`) for testing other available functionalities.

See other files in the `tests` and `demos` directories for more usage examples.


