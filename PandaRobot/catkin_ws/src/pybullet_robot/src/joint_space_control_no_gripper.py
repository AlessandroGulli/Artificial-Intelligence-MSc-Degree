import sys,os
import numpy as np
import pybullet as pb
sys.path.append("./pybullet_robot")
from pybullet_robot.worlds import SimpleWorld, add_PyB_models_to_path
from pybullet_robot.robots import PandaArm
from pybullet_robot.controllers import OSImpedanceController, OSImpedanceControllerJointSpace
from pybullet_robot.controllers.utils import execute_joints_trajectory, plan_pos_trajectory, path_planning, plan_ori_trajectory, quaternion_to_euler_angle, display_trajectories  
import time



if __name__ == "__main__":
    robot = PandaArm()

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

    world = SimpleWorld(robot, objects)
    slow_rate = 500.
    init_pos, init_ori = world.robot.ee_pose()   
    controller = OSImpedanceControllerJointSpace(robot)
 
    print("started")
    controller.start_controller_thread()
     
    T = 3 #s
    NPoints = T*slow_rate # 2*200 
    
    ##Stay
    p_x_e_i = np.copy(init_pos)
    p_x_e_f = np.copy(init_pos)
    trajectory_0_pos = np.concatenate([p_x_e_i,p_x_e_f])    
    o_x_e_i = quaternion_to_euler_angle(init_ori.w, init_ori.x, init_ori.y, init_ori.z)
    o_x_e_f = np.copy(o_x_e_i)
    trajectory_0_ori = np.concatenate([o_x_e_i,o_x_e_f])
    
    grip_cmd = -1
    trajectory_0 = np.concatenate([trajectory_0_pos,trajectory_0_ori])    
    x_e_0, dx_e_0, g0, t0 = path_planning(trajectory_0[0:3], trajectory_0[3:6], trajectory_0[6:9], trajectory_0[9:12], T, NPoints, grip_cmd)
    
    ##Down
    p_x_e_i = np.copy(init_pos) 
    p_x_e_f = [p_x_e_i[0], p_x_e_i[1], 0.46] 
    trajectory_1_pos = np.concatenate([p_x_e_i,p_x_e_f])    
    o_x_e_i = quaternion_to_euler_angle(init_ori.w, init_ori.x, init_ori.y, init_ori.z)
    o_x_e_f = [o_x_e_i[0], o_x_e_i[1], -np.pi/2]
    trajectory_1_ori = np.concatenate([o_x_e_i,o_x_e_f])
    
    grip_cmd = 0
    trajectory_1 = np.concatenate([trajectory_1_pos,trajectory_1_ori])    
    x_e_1, dx_e_1, g1, t1 = path_planning(trajectory_1[0:3], trajectory_1[3:6], trajectory_1[6:9], trajectory_1[9:12], T, NPoints, grip_cmd)   
    
    ##Stay
    p_x_e_i = np.copy(p_x_e_f)
    p_x_e_f = np.copy(p_x_e_i)
    trajectory_2_pos = np.concatenate([p_x_e_i,p_x_e_f])    
    o_x_e_i = np.copy(o_x_e_f)
    o_x_e_f = [o_x_e_i[0], np.pi/7, o_x_e_i[2]]
    trajectory_2_ori = np.concatenate([o_x_e_i,o_x_e_f])
    
    grip_cmd = 0
    trajectory_2 = np.concatenate([trajectory_2_pos,trajectory_2_ori])    
    x_e_2, dx_e_2, g2, t2 = path_planning(trajectory_2[0:3], trajectory_2[3:6], trajectory_2[6:9], trajectory_2[9:12], T, NPoints, grip_cmd)

    ##Up
    p_x_e_i = np.copy(p_x_e_f)
    p_x_e_f = [p_x_e_i[0], 0.35, 0.5]
    trajectory_3_pos = np.concatenate([p_x_e_i,p_x_e_f])    
    o_x_e_i = np.copy(o_x_e_f)
    o_x_e_f = np.copy(o_x_e_f)
    trajectory_3_ori = np.concatenate([o_x_e_i,o_x_e_f])
    
    grip_cmd = 0
    trajectory_3 = np.concatenate([trajectory_3_pos,trajectory_3_ori])    
    x_e_3, dx_e_3, g3, t3 = path_planning(trajectory_3[0:3], trajectory_3[3:6], trajectory_3[6:9], trajectory_3[9:12], T, NPoints, grip_cmd) 

    x_e = np.concatenate([x_e_0, x_e_1, x_e_2, x_e_3])
    dx_e = np.concatenate([dx_e_0, dx_e_1, dx_e_2, dx_e_3])
    t = np.concatenate([t0, t0[-1] + t1, t0[-1] + t1[-1] + t2, t0[-1] + t1[-1] + t2[-1] + t3])
    g = np.concatenate([g0, g1, g2, g3])
     	 
    sim_outcome = execute_joints_trajectory(x_e, dx_e, g, world, controller, slow_rate)
    
    controller.stop_controller_thread()
    time.sleep(1)
    print("stopped") 
        
    display_trajectories(sim_outcome, x_e, dx_e, t)
    
