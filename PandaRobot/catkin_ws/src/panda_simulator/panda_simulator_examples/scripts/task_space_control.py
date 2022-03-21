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
sys.path.append("../../pybullet_robot")
import pybullet as pb
from pybullet_robot.controllers import OSImpedanceController, OSImpedanceControllerJointSpace
from pybullet_robot.controllers.utils import euler_to_quaternion_raw, quaternion_to_euler_angle
from pybullet_robot.worlds import SimpleWorld, add_PyB_models_to_path
from pybullet_robot.robots import PandaArm
from pybullet_robot.controllers.utils import display_trajectories  
from pybullet_robot.controllers.planning import Trajectory_Generator
from pybullet_robot.controllers.traj_config_operational import Traj_Config

# -------------------------------------------------

from rviz_markers import RvizMarkers

# --------- Modify as required ------------
# Task-space controller parameters
# stiffness gains
P_pos = 50.
P_ori = 25.
# damping gains
D_pos = 1.
D_ori = 1.
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
                                   
                controller.update_goal(planning._x_e[i][0:3], euler_to_quaternion_raw(pb, planning._x_e[i][3:6]))
                # PyBullet        

                # when using the panda_robot interface, the next 2 lines can be simplified 
                # to: `curr_pos, curr_ori = panda.ee_pose()`
                curr_pos = robot.endpoint_pose()['position']
                curr_ori = np.asarray(robot.endpoint_pose()['orientation'])

                goal_pos = planning._x_e[i][0:3]
                delta_pos = (goal_pos - curr_pos).reshape([3,1])
                
                goal_ori = euler_to_quaternion_raw(pb, planning._x_e[i][3:6])
                goal_ori = np.quaternion(goal_ori[3], goal_ori[0], goal_ori[1], goal_ori[2])
                delta_ori = quatdiff_in_euler(curr_ori, goal_ori).reshape([3,1])

                # when using the panda_robot interface, the next 2 lines can be simplified 
                # to: `curr_vel, curr_omg = panda.ee_velocity()
                curr_vel = robot.endpoint_velocity()['linear'].reshape([3,1])
                curr_omg = robot.endpoint_velocity()['angular'].reshape([3,1])

                # Desired task-space force using PD law
                F = np.vstack([P_pos*(delta_pos), P_ori*(delta_ori)]) - \
                    np.vstack([D_pos*(curr_vel), D_ori*(curr_omg)])

                error = np.linalg.norm(delta_pos) + np.linalg.norm(delta_ori)
            
                # panda_robot equivalent: panda.jacobian(angles[optional]) or panda.zero_jacobian()
                J = robot.zero_jacobian()
            
                # joint torques to be commanded
                tau = np.dot(J.T,F)

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
        
    controller = OSImpedanceController(bullet_robot)
 
    planning = Trajectory_Generator(Traj_Config)
    planning.path_assembly()     
    
    rospy.on_shutdown(_on_shutdown)
    rate = rospy.Rate(publish_rate)
    controller.start_controller_thread()
    ctrl_thread = threading.Thread(target=control_thread, args = [bullet_robot, rate, controller, planning, pb, gripper])
    ctrl_thread.start()  
    
    rospy.spin()
 
    
    
