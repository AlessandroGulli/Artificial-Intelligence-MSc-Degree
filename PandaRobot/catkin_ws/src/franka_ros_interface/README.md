# Franka ROS Interface [![Release](https://img.shields.io/badge/release-v0.7.1-blue.svg)](https://github.com/justagist/franka_ros_interface/tree/v0.7.1-dev) [![ROS Version](https://img.shields.io/badge/ROS-Melodic,%20Noetic-brightgreen.svg?logo=ros)](https://ros.org/) [![Python 2.7+, 3.6+](https://img.shields.io/badge/python-2.7+,%203.6+-blue.svg?logo=python)](https://www.python.org/downloads/release/python-360/)

A ROS interface library for the Franka Emika Panda robot (**_real and [simulated][ps-repo]_**), extending the [franka-ros][franka-ros] library to expose more information about the robot, and
providing low-level control of the robot using ROS and [Python API][fri-doc].

Franka ROS Interface provides utilites for controlling and managing the Franka Emika Panda robot. Contains exposed customisable controllers for the robot (joint position,
velocity, torque), interfaces for the gripper, controller manager, coordinate frames interface, etc. Also provides utilities to control the robot using 'MoveIt!' and ROS Trajectory Action & ActionClient. This package also provides almost complete sim-to-real / real-to-sim transfer of code with the [Panda Simulator][ps-repo] package.

Documentation Page: <https://justagist.github.io/franka_ros_interface>

**This branch requires franka_ros release version 0.7.1** [![franka_ros_version](https://img.shields.io/badge/franka_ros-v0.7.1%20release-yellow.svg)](https://github.com/frankaemika/franka_ros/tree/49e5ac1055e332581b4520a1bd9ac8aaf4580fb1). (For older franka_ros versions, try building this package from the corresponding branches of this repo. All functionalities may not be available in older versions.)

A more unified ROS Python interface built over this package is available at [PandaRobot](https://github.com/justagist/panda_robot), which provides a more intuitive interface class that combines the different API classes in this package. Simple demos are also available.



**More usage examples can be found in the [PandaRobot](https://github.com/justagist/panda_robot) package.**

#### Some useful ROS topics

##### Published Topics

| ROS Topic | Data |
| ------ | ------ |
| */franka_ros_interface/custom_franka_state_controller/robot_state* | gravity, coriolis, jacobian, cartesian velocity, etc. |
| */franka_ros_interface/custom_franka_state_controller/tip_state* | end-effector pose, wrench, etc. |
| */franka_ros_interface/joint_states* | joint positions, velocities, efforts |
| */franka_ros_interface/franka_gripper/joint_states* | joint positions, velocities, efforts of gripper joints |

##### Subscribed Topics

| ROS Topic | Data |
| ------ | ------ |
| */franka_ros_interface/motion_controller/arm/joint_commands* | command the robot using the currently active controller |
| */franka_ros_interface/franka_gripper/[move/grasp/stop/homing]* | (action msg) command the joints of the gripper |

Other topics for changing the controller gains (also dynamically configurable), command timeout, etc. are also available.

#### ROS Services

Controller manager service can be used to switch between all available controllers (joint position, velocity, effort). Gripper joints can be controlled using the ROS ActionClient. Other services for changing coordinate frames, adding gripper load configuration, etc. are also available.

#### Python API

Most of the above services and topics are wrapped using simple Python classes or utility functions, providing more control and simplicity. This includes direct control of the robot and gripper using the provided custom low-level controllers, MoveIt, and JointTrajectoryAction. Refer README files in individual subpackages.

## Related Packages

- [*panda_simulator*][ps-repo] : A Gazebo simulator for the Franka Emika Panda robot with ROS interface, providing exposed controllers and real-time robot state feedback similar to the real robot when using the *franka_ros_interface* package. Provides almost complete real-to-sim transfer of code.
- [*PandaRobot*](https://github.com/justagist/panda_robot) : Python interface providing higher-level control of the robot integrated with its gripper control, controller manager, coordinate frames manager, etc. with safety checks and other helper utilities. It also provides the kinematics and dynamics of the robot using the [KDL library](http://wiki.ros.org/kdl). It is built over Franka ROS Interface and provides a more intuitive and unified single-class interface.
- [*franka_panda_description*][fpd-repo] : Robot description package modified from [*franka_ros*][franka-ros] package to include dynamics parameters for the robot arm (as estimated in [this paper](https://hal.inria.fr/hal-02265293/document)). Also includes transmission and control definitions required for the [*panda_simulator*][ps-repo] package.

   [ps-repo]: <https://github.com/justagist/panda_simulator>
   [fri-repo]: <https://github.com/justagist/franka_ros_interface>
   [fpd-repo]: <https://github.com/justagist/franka_panda_description>
   [fri-doc]: <https://justagist.github.io/franka_ros_interface>
   [libfranka-doc]: <https://frankaemika.github.io/docs/installation_linux.html#building-from-source>
   [franka-ros]: <https://github.com/frankaemika/franka_ros>

