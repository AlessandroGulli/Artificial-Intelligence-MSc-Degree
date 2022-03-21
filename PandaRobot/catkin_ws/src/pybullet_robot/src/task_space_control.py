#!/usr/bin/env python
import sys,os
import numpy as np
import pybullet as pb
sys.path.append("./pybullet_robot")
from pybullet_robot.worlds import SimpleWorld, add_PyB_models_to_path
from pybullet_robot.robots import PandaArm
from pybullet_robot.controllers import OSImpedanceController
from pybullet_robot.controllers.utils import display_trajectories
from pybullet_robot.controllers.planning import Trajectory_Generator  
from pybullet_robot.controllers.traj_config_operational import Traj_Config
import time

def task_control():

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
    controller = OSImpedanceController(robot)
    
    pb.configureDebugVisualizer(pb.COV_ENABLE_RENDERING,1)
 
    planning = Trajectory_Generator(Traj_Config)
    planning.path_assembly()
    pb.setTimeStep(1./planning._rate)        

    controller.start_controller_thread()

    sim_outcome = planning.execute_task_trajectory(planning._x_e, planning._dx_e, planning._g, world, controller, planning._rate, pb)

    controller.stop_controller_thread()    

    display_trajectories(sim_outcome, planning._x_e, planning._dx_e, planning._t)
    
    exit()


if __name__ == "__main__":
    task_control()
    
