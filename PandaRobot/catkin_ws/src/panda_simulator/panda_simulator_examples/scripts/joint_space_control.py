#!/usr/bin/env python3

import copy
import rospy
import threading
import quaternion
import time
import numpy as np
from geometry_msgs.msg import Point
from visualization_msgs.msg import *
from interactive_markers.interactive_marker_server import *
from franka_interface import ArmInterface
from franka_interface import GripperInterface
from franka_core_msgs.msg import JointCommand

# -- add to pythonpath for finding rviz_markers.py 
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append("../pybullet_robot")
import pybullet as pb
from pybullet_robot.controllers import OSImpedanceController, OSImpedanceControllerJointSpace
from pybullet_robot.controllers.utils import euler_to_quaternion_raw, quaternion_to_euler_angle
from pybullet_robot.worlds import SimpleWorld, add_PyB_models_to_path
from pybullet_robot.robots import PandaArm
from pybullet_robot.controllers.utils import display_trajectories  
from pybullet_robot.controllers.planning import Trajectory_Generator
from pybullet_robot.controllers.traj_config_joints import Traj_Config

# -------------------------------------------------

from rviz_markers import RvizMarkers

# --------- Modify as required ------------
# Joint-space controller parameters
# stiffness gains
kP = np.asarray([10., 10., 10., 10., 10., 10., 10.]) 
# damping gains  
kD = np.asarray([1. , 1. , 1. , 1. , 1. , 1. , 1.])


# -----------------------------------------
publish_rate = 500

JACOBIAN = None
CARTESIAN_POSE = None
CARTESIAN_VEL = None

destination_marker = RvizMarkers()


def home_position(joint_config):
     
    pub = rospy.Publisher('panda_simulator/motion_controller/arm/joint_commands',JointCommand,tcp_nodelay=True,queue_size=10)

    command_msg = JointCommand()
    command_msg.names = ["panda_joint%d" % (idx) for idx in range(1, 8)]     
    command_msg.position = joint_config
    command_msg.mode = JointCommand.POSITION_MODE
    
    rospy.sleep(0.5)
    start = rospy.Time.now().to_sec()

    rospy.loginfo("Attempting to force robot to neutral pose...")
    rospy.sleep(0.5)
    
    while not rospy.is_shutdown() and (rospy.Time.now().to_sec() - start < 1.):
        # print rospy.Time.now()
        command_msg.header.stamp = rospy.Time.now()
        pub.publish(command_msg)

    rospy.loginfo("Robot forced to neutral pose. Complete!")
    	
    return 


def quatdiff_in_euler(quat_curr, quat_des):
    """
        Compute difference between quaternions and return 
        Euler angles as difference
    """
    curr_mat = quaternion.as_rotation_matrix(quat_curr)
    des_mat = quaternion.as_rotation_matrix(quat_des)
    rel_mat = des_mat.T.dot(curr_mat)
    rel_quat = quaternion.from_rotation_matrix(rel_mat)
    vec = quaternion.as_float_array(rel_quat)[1:]
    if rel_quat.w < 0.0:
        vec = -vec
        
    return -des_mat.dot(vec)

def control_thread(bullet_robot, rate, controller, planning, pb, gripper):
    """
        Actual control loop. Uses goal pose from the feedback thread
        and current robot states from the subscribed messages to compute
        task-space force, and then the corresponding joint torques.
    """
    i = 0   
    while (not rospy.is_shutdown()) and (i < len(planning._x_e)):  
                
                # PyBullet           
                if planning._g[i,[0]] == 1:
                     bullet_robot.gripper_open(pos = planning._g[i,[1]], force = planning._g[i,[2]]) 
                     gripper.open() #--> ROS                    
                elif planning._g[i,[0]] == -1:
                     bullet_robot.gripper_close(pos = planning._g[i,[1]], force= planning._g[i,[2]]) 
                     gripper.close() #--> ROS
                                   
                controller.update_goal(planning._x_e[i][0:3], euler_to_quaternion_raw(pb, planning._x_e[i][3:6]), planning._dx_e[i][0:3], planning._dx_e[i][3:6])
                # PyBullet 
                
                goal_pos = planning._x_e[i][0:3]
                goal_ori = euler_to_quaternion_raw(pb, planning._x_e[i][3:6])
                goal_vel = planning._dx_e[i][0:3]
                goal_omg = planning._dx_e[i][3:6]
                goal_whole_joint_angles = bullet_robot.position_ik(goal_pos, goal_ori)
                goal_joint_angles = goal_whole_joint_angles[0:7]
                goal_joint_velocities = np.dot(np.linalg.pinv((bullet_robot.jacobian(goal_whole_joint_angles)[:, 0:7])) , np.array([np.concatenate([goal_vel , goal_omg])]).T).flatten()

                curr_joint_angles = robot.joint_ordered_angles()
                curr_joint_velocities = robot.joint_ordered_velocities()
  
                delta_angles     = goal_joint_angles     - curr_joint_angles
                delta_velocities = goal_joint_velocities - curr_joint_velocities
                       
                # joint torques to be commanded
                tau = np.add(kP*(delta_angles).T, kD*(delta_velocities).T) 

                error = np.asarray([np.linalg.norm(delta_angles), np.linalg.norm(delta_velocities)])

                # command robot using joint torques
                # panda_robot equivalent: panda.exec_torque_cmd(tau)
                robot.set_joint_torques(dict(list(zip(robot.joint_names(), tau))))

                i +=1
                rate.sleep()  
                
    controller.stop_controller_thread()      
    return 0                           


def _on_shutdown():
    """
        Clean shutdown controller thread when rosnode dies.
    """
    global ctrl_thread
    if ctrl_thread.is_alive():
        ctrl_thread.join()
    
if __name__ == "__main__":

   
    bullet_robot = PandaArm(uid="DIRECT")

    add_PyB_models_to_path()

    plane = pb.loadURDF('plane.urdf')
    table = pb.loadURDF('table/table.urdf',
                        useFixedBase=True, globalScaling=0.5)                      
    cube = pb.loadURDF('cube_small.urdf', globalScaling=1.)
    
    pb.resetBasePositionAndOrientation(table, [0.4, -0.8, 0.0], [0, 0, -0.707, 0.707])
    
    red    = [0.97, 0.25, 0.25, 1]
    green  = [0.41, 0.68, 0.31, 1]
    yellow = [0.92, 0.73, 0, 1]
    blue   = [0, 0.55, 0.81, 1] 
       
    pb.changeVisualShape(cube,-1,rgbaColor=red)   
    pb.resetBasePositionAndOrientation(cube, [0.4, 0.5, 0.5], [0, 0, -0.707, 0.707])
    
    objects = {'plane': plane, 
               'table': table,              
               'cube': cube}         

    world = SimpleWorld(bullet_robot, objects) 
    state = world.robot.state()  
    init_angles = state['position'][0:8]
    
    #### Start ROS node ####
    rospy.init_node("Task Space Control")
    #### Start ROS node ####
    
    robot = ArmInterface()
    gripper = GripperInterface()

    #Home Position
    home_position(init_angles)
    gripper.close()     
    
    rospy.on_shutdown(_on_shutdown)
    rate = rospy.Rate(publish_rate)
        
    controller = OSImpedanceControllerJointSpace(bullet_robot)
 
    planning = Trajectory_Generator(Traj_Config)
    planning.path_assembly()     
    
    rospy.on_shutdown(_on_shutdown)
    rate = rospy.Rate(publish_rate)
    controller.start_controller_thread()
    ctrl_thread = threading.Thread(target=control_thread, args = [bullet_robot, rate, controller, planning, pb, gripper])
    ctrl_thread.start()  
    
    rospy.spin()
 
    
    
