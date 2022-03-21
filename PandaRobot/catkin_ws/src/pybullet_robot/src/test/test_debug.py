#!/usr/bin/env python
import sys,os
sys.path.append('./pybullet_robot')
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import pybullet as pb
import matplotlib.pyplot as plt
from pybullet_robot.worlds import SimpleWorld, add_PyB_models_to_path
from pybullet_robot.robots import PandaArm
import quaternion
from pybullet_robot.controllers.utils import euler_to_quaternion_raw, quatdiff_in_euler, weighted_minkowskian_distance, calc_reference_frames
from pybullet_robot.controllers import OSImpedanceController, OSImpedanceControllerJointSpace
from pybullet_robot.controllers.utils import display_trajectories
from pybullet_robot.controllers.planning import Trajectory_Generator  
from pybullet_robot.controllers.traj_config_joints_training import Traj_Config
import time
import pybullet_data
import math
import numpy as np
import random
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import StopTrainingOnMaxEpisodes
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CallbackList
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
import multiprocessing as mp
import datetime


class PandaEnv(gym.Env):

    metadata = {'render.modes':['human']}

    def __init__(self, render_enable = False, maxSteps = 10000): 
        super(PandaEnv, self).__init__()
 
        self.renders = render_enable
        self.robot = PandaArm()       
        self.planning = Trajectory_Generator(Traj_Config)        
                       
        add_PyB_models_to_path()              

        # Action space
        # q_dot_target -> 7 items, one for each joint 
        n_joints = 7
        total_actions = n_joints            
        self.action_space = spaces.Box(np.array([-1]*total_actions), np.array([1]*total_actions))

        # Observation space
        # delta_x -> 6 items, 3 for position and 3 for orientation 
        delta_cartesian_pos_ori = 6
        # q       -> 7 items, one for each joint
        joint_angles = 7
        # q_dot   -> 7 items, one for each joint
        joint_velocities = 7

        total_observations = delta_cartesian_pos_ori + joint_angles + joint_velocities
        #self.observation_space = spaces.Box(np.array([-np.inf]*total_observations), np.array([np.inf]*total_observations))
        self.observation_space = spaces.Box(np.array([-1]*total_observations), np.array([1]*total_observations))

        self.maxSteps = maxSteps
                        
        self.window_size = 10           
        self.sampled_values = np.zeros((self.window_size, 7))

        self.i = 0   
        self.avg_idx = 0 
        
        self.joint_velocities_display = list()
        self.joint_accelerations_display = list()
        self.reward_stored = list()
        self.joint_accelerations_filtered_display = list()
        self.total_joint_accelerations_display = list()
        self.angles_display = np.asarray(self.robot.angles()).reshape([9,1])
        self.velocity_display = np.asarray(self.robot.joint_velocities()).reshape([9,1])
        
 
    def reset(self):
                            
        self.robot.reset() 

        add_PyB_models_to_path()  

        plane = pb.loadURDF('plane.urdf')
        table = pb.loadURDF('table/table.urdf', useFixedBase=True, globalScaling=0.5)                      
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
                   'cube' : cube}           
       
        self.world = SimpleWorld(self.robot, objects)         
        self.robot.set_ctrl_mode('tor')
        self.world.robot.gripper_close(pos = 0., force= 0.)
        
        pb.configureDebugVisualizer(pb.COV_ENABLE_RENDERING,1) # rendering's back on again

        curr_pos, curr_ori = self.robot.ee_pose()
        goal_ori = euler_to_quaternion_raw(pb, self.planning._points[2][3:6])
        goal_ori = np.quaternion(goal_ori[3], goal_ori[0], goal_ori[1], goal_ori[2]) 
        delta_pos = np.asarray(self.planning._points[2][0:3]).reshape([3,1]) - curr_pos.reshape([3,1])      
        delta_ori = quatdiff_in_euler(curr_ori, goal_ori).reshape([3,1])  
        delta_x = np.concatenate([delta_pos, delta_ori])  
        
        self.goal_joint_angles = np.asarray(self.robot.angles()).reshape([9,1])
        self.goal_joint_velocities = np.asarray(self.robot.joint_velocities()).reshape([9,1])
        self.prev_joint_velocities = np.asarray(self.robot.joint_velocities()).reshape([9,1])

        self.start_time = time.time()

        observation = np.concatenate([ delta_x.reshape([6,1]), self.goal_joint_angles[0:7].reshape([7,1]), self.goal_joint_velocities[0:7].reshape([7,1]) ]).reshape(-1)   
        
        self.joint_velocities_display.clear()
        self.joint_accelerations_display.clear()
        self.joint_accelerations_filtered_display.clear()
        self.reward_stored.clear()
        self.total_joint_accelerations_display.clear()
        self.angles_display = np.asarray(self.robot.angles()).reshape([9,1])
        self.velocity_display = np.asarray(self.robot.joint_velocities()).reshape([9,1])
        
        self.avg_idx = 0 
        self.sampled_values = np.zeros((self.window_size, 7))   

        return observation

    def step(self, action):

        now = time.time()  
        
        action = np.concatenate([action, [0., 0.]])
        
        tmp_goal_joint_velocities_normalized = np.asarray(action).reshape([9,1])        
        limits = np.asarray(self.robot.get_joint_velocity_limits()).reshape([9,1])
        #normalize back to controller        
        tmp_goal_joint_velocities = limits*tmp_goal_joint_velocities_normalized
        tmp_goal_joint_angles = np.add(self.goal_joint_angles, (1./self.planning._rate)*tmp_goal_joint_velocities)
        tmp_goal_joint_angles = np.asarray(np.clip(tmp_goal_joint_angles.reshape(-1), a_min=self.robot.get_joint_limits()['lower'], a_max=self.robot.get_joint_limits()['upper'])).reshape([9, 1])
        
        self.angles_display = np.hstack((self.angles_display, tmp_goal_joint_angles))
        self.velocity_display = np.hstack((self.velocity_display, tmp_goal_joint_velocities))

        """
        Actual control loop. Uses goal pose from the feedback thread
        and current robot states from the subscribed messages to compute
        task-space force, and then the corresponding joint torques.
        """
          
        curr_joint_angles =  np.asarray(self.robot.angles()).reshape([9,1])
        curr_joint_velocities =  np.asarray(self.robot.joint_velocities()).reshape([9,1])
        #joint accelerations
        curr_joint_accelerations = (curr_joint_velocities - self.prev_joint_velocities)*self.planning._rate
        self.prev_joint_velocities = curr_joint_velocities
              
        delta_angles     = tmp_goal_joint_angles - curr_joint_angles
        delta_velocities = tmp_goal_joint_velocities - curr_joint_velocities

        # Desired joint-space torque using PD law
        kP = np.asarray([1000., 1000., 1000., 1000., 1000., 1000., 500., 0., 0.])    
        kD =  np.asarray([2.   , 2.   , 2.  , 2.  , 2.  , 2. , 1. , 0., 0.])
        error_thresh = np.asarray([0.010, 0.010])
    
        tau = np.add(kP*(delta_angles).T, kD*(delta_velocities).T).reshape(-1) 

        error = np.asarray([np.linalg.norm(delta_angles), np.linalg.norm(delta_velocities)])
        
        # joint torques to be commanded
        torque_cmd = tau + self.robot.torque_compensation()

        total_time = time.time() - self.start_time              

        self.goal_joint_angles = tmp_goal_joint_angles
        
        #Check respect to final pose
        curr_pos, curr_ori = self.robot.ee_pose()
        goal_ori = euler_to_quaternion_raw(pb, self.planning._points[2][3:6])
        goal_ori = np.quaternion(goal_ori[3], goal_ori[0], goal_ori[1], goal_ori[2]) 
        delta_pos = np.asarray(self.planning._points[2][0:3]).reshape([3, 1]) - curr_pos.reshape([3, 1])       
        delta_ori = quatdiff_in_euler(curr_ori, goal_ori).reshape([3, 1])  
        delta_x = np.concatenate([delta_pos, delta_ori])

        err_pos = np.linalg.norm(delta_pos)
        err_ori = np.linalg.norm(delta_ori)
        global_target_error = np.asarray([err_pos, err_ori])
        
        #print(self.planning._points[2][0:3], curr_pos, err_pos)
        #print(f"Target Pos: {self.planning._points[2][0:3]}\nCurr Pos: {curr_pos.reshape([3, 1])}\nError Position:{err_pos}\nError Orientation:{err_ori}")

        elapsed_time = time.time() - now
        sleep_time = (1./self.planning._rate) - elapsed_time        
        if sleep_time > 0.0:
            time.sleep(sleep_time)

        if np.any(global_target_error > error_thresh):            
            self.robot.exec_torque_cmd(torque_cmd)
            self.robot.step_if_not_rtsim()
            if total_time > 30 or np.any(err_pos > 0.8):
                done = True 
                print(f"Time is over") 
                
            else:
                done = False
               
        elif np.all(global_target_error < error_thresh):
            done = True            
            now = datetime.datetime.now()
            print(f"Enviroment resolved on {now.day}/{now.month}/{now.year} at {now.hour}:{now.minute}:{now.second}")      
        
        observation = np.concatenate([ delta_x.reshape([6,1]), self.robot.angles()[0:7].reshape([7,1]), self.robot.joint_velocities()[0:7].reshape([7,1]) ]).reshape(-1) 
        
        self.joint_velocities_display.append(self.robot.joint_velocities()[0:7])
        self.joint_accelerations_display.append((self.robot.joint_velocities()[0:7][0:7]*self.planning._rate) if len(self.joint_accelerations_display) == 0 else ((self.robot.joint_velocities()[0:7][0:7] - self.joint_velocities_display[-2])*self.planning._rate))
        
        
        if self.avg_idx > (self.window_size - 1):
                self.avg_idx = 0
        self.sampled_values[self.avg_idx] = self.joint_accelerations_display[-1]            
        self.avg_idx +=1            
        avg_values = np.mean(self.sampled_values, axis=0)          
            
        self.joint_accelerations_filtered_display.append(avg_values)
        self.total_joint_accelerations_display.append(np.linalg.norm(avg_values))           
            
        X, Y, Z, X_target, Y_target, Z_target = calc_reference_frames(self.world, self.planning._points[2][0:3], self.planning._points[2][3:6], pb)
        
        delta_x = np.asarray(X_target.reshape(3,1) - X.reshape(3,1))
        err_x = np.linalg.norm(delta_x)            
        delta_y = np.asarray(Y_target.reshape(3,1) - Y.reshape(3,1))
        err_y = np.linalg.norm(delta_y)            
        delta_z = np.asarray(Z_target.reshape(3,1) - Z.reshape(3,1))            
        err_z = np.linalg.norm(delta_z)            
                
        lamba_err = 0.8
        lamba_eff = 0.001        
        delta_x_error = err_x + err_y + err_z
        acc_error = np.linalg.norm(self.total_joint_accelerations_display[-1])
        reward = np.exp(-lamba_err*np.square(delta_x_error)) - np.clip(lamba_eff*acc_error, 0, 1)
        self.reward_stored.append(reward)
        info = {} 
        
        if done == True:
        
                
                fig2 = plt.figure(figsize=(20,10), constrained_layout=True)    
                ax5 = fig2.add_subplot(211)
                ax6 = fig2.add_subplot(212)  
    
                ax5.set_title("Joint Angles")    
                #ax5.plot(self.angles_display[0, :], 'r-', label ='J0')
                #ax5.plot(self.angles_display[1, :], 'g-', label ='J1')
                #ax5.plot(self.angles_display[2, :], 'b-', label ='J2')
                ax5.plot(self.angles_display[3, :], 'c-', label ='J3')
                ax5.plot(self.angles_display[4, :], 'm-', label ='J4')
                #ax5.plot(self.angles_display[5, :], 'y-', label ='J5') 
                #ax5.plot(self.angles_display[6, :], 'k-', label ='J6') 
                ax5.set_xlabel("time [s]")   
                ax5.set_ylabel("q(t) [rad]") 
                ax5.legend()  
   
                ax6.set_title("Joint Velocities")  
                #ax6.plot(self.velocity_display[0, :], 'r-', label ='J0')
                #ax6.plot(self.velocity_display[1, :], 'g-', label ='J1')
                #ax6.plot(self.velocity_display[2, :], 'b-', label ='J2')
                ax6.plot(self.velocity_display[3, :], 'c-', label ='J3')
                ax6.plot(self.velocity_display[4, :], 'm-', label ='J4')
                #ax6.plot(self.velocity_display[5, :], 'y-', label ='J5') 
                #ax6.plot(self.velocity_display[6, :], 'k-', label ='J6') 
                ax6.set_xlabel("time [s]")   
                ax6.set_ylabel("dq(t) [rad/s]")
                ax6.legend()  
                
                
                fig4 = plt.figure(figsize=(20,10), constrained_layout=True)    
                ax9 = fig4.add_subplot(211)
                ax10 = fig4.add_subplot(212)
    
                ax9.set_title("Total Joints Acceleration")  
                ax9.plot(self.total_joint_accelerations_display, 'r-')
                ax9.set_xlabel("time [s]")   
                ax9.set_ylabel("ddq(t) [rad/s^2]")
                ax9.legend()  
    
                ax10.set_title("Reward")  
                ax10.plot(self.reward_stored, 'c-')
                ax10.set_xlabel("time [s]")   
                ax10.legend()
                
                plt.show()
        
        return observation, reward, done, info


    def render(self, mode='human'):
        view_matrix = pb.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0.7,0,0.05], distance=.7, yaw=90, pitch=-70, roll=0, upAxisIndex=2)                              
        proj_matrix = pb.computeProjectionMatrixFOV(fov=60, aspect=float(960) /720, nearVal=0.1, farVal=100.0)
        (_, _, px, _, _) = pb.getCameraImage(width=960, height=720, viewMatrix=view_matrix, projectionMatrix=proj_matrix, renderer=pb.ER_BULLET_HARDWARE_OPENGL)

        rgb_array = np.array(px, dtype=np.uint8)
        rgb_array = np.reshape(rgb_array, (720,960, 4))
        rgb_array = rgb_array[:, :, :3]
                
        return rgb_array

    def close(self):
        self.robot.__del__        

if __name__ == '__main__':
   
    # Create log dir
    log_dir = './RL/models/archive/311221_test/'
    os.makedirs(log_dir, exist_ok=True) 
    stats_path = os.path.join(log_dir, "vec_normalize.pkl")   

    env = PandaEnv()        
            
    env = DummyVecEnv([lambda: env])        
    # Automatically normalize the input features and reward
    env = VecNormalize.load(stats_path, env) 
    env.training = False 
    env.norm_reward = False
    
    model = PPO.load(log_dir + "best_model", env=env)

    done = False
    t = 0
    for i_episode in range(3):
        observation = env.reset() 
    
        for t in range(1000):
            env.render()        
            action, states = model.predict(observation)#env.action_space.sample()
                    
            observation, reward, done, info = env.step(action)

            #print(f'Observations: {observation}\nReward: {reward}\nDone: {done}\nInfo: {info}')
 
            if done == True:
                print(f"Episode finished after {t+1} timesteps")                
                break
 
    env.close()
    
	
