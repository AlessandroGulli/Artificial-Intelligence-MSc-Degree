import quaternion
import numpy as np
import time
import matplotlib.pyplot as plt
from pybullet_robot.controllers.utils import euler_to_quaternion_raw, quaternion_to_euler_angle, butter_lowpass_filter, calc_reference_frames

class Trajectory_Generator(object):

    def __init__(self, trajectory):
            
        self._points = trajectory['points']
        self._T = trajectory['intervals']
        self._NPoints = trajectory['NPoints']
        self._gripper_cmd = trajectory['gripper_cmd']
        self._rate = trajectory['rate']
        
    def polynomial_3_mod(self, T, NPoints, Q0):

        f_0  = Q0[0]
        df_0 = Q0[1]
        f_T  = Q0[2]
        df_T = Q0[3]
        
        a_0 = f_0
        a_1 = df_0
        a_2 = (3*(f_T - f_0) - T*(2*df_0 + df_T))/(T*T)
        a_3 = -(2*(f_T - f_0) - T*(df_0 + df_T))/(T*T*T)    
           
        p = np.linspace(0, T, round(int(NPoints)))    
        f = a_0 + a_1*p + a_2*p*p + a_3*p*p*p
        df = a_1 + 2*a_2*p + 3*a_3*p*p
        ddf = 2*a_2 + 6*a_3*p
        dddf = 6*a_3*np.ones((1,len(ddf)))

        F = [f, df, ddf, dddf] 

        return p,F
    
    def plan_pos_trajectory(self, x_e_i, x_e_f, T, NPoints):

        ### Position/Velocity
        s_pos_start = 0
        s_pos_end = np.linalg.norm(x_e_f - x_e_i) 
        p, s_pos = self.polynomial_3_mod(T, NPoints, [s_pos_start, 0, s_pos_end, 0])
       
        x_e = np.zeros((len(s_pos[0]),6)) 
        dx_e = np.zeros((len(s_pos[1]),6)) 

        for i in range(len(s_pos[0])):
            if any(x_e_f - x_e_i) != 0:
                x_e[i][0:3] = x_e_i + (s_pos[0][i]/np.linalg.norm(x_e_f - x_e_i))*(x_e_f - x_e_i)
                dx_e[i][0:3] = (s_pos[1][i]/np.linalg.norm(x_e_f - x_e_i))*(x_e_f - x_e_i)
            else:
                x_e[i][0:3] = x_e_i
                dx_e[i][0:3] = 0      
        ### Position/Velocity    

        return x_e, dx_e, p 


    def plan_ori_trajectory(self, x_e, dx_e, x_e_i, x_e_f, T, NPoints):
         
        ### Orientation/Angular Velocity
        s_pos_start = 0
        s_pos_end = np.linalg.norm(x_e_f - x_e_i) 
        p, s_pos = self.polynomial_3_mod(T, NPoints, [s_pos_start, 0, s_pos_end, 0])

        for i in range(len(s_pos[0])):
            if any(x_e_f - x_e_i) != 0:
                x_e[i][3:6] = x_e_i + (s_pos[0][i]/np.linalg.norm(x_e_f - x_e_i))*(x_e_f - x_e_i)
                dx_e[i][3:6] = (s_pos[1][i]/np.linalg.norm(x_e_f - x_e_i))*(x_e_f - x_e_i)        	
            else:
                x_e[i][3:6] = x_e_i
                dx_e[i][3:6] = 0
        ### Orientation/Angular Velocity   

        return x_e, dx_e, p		
        
    def path_planning(self, init_pos, final_pos, init_ori, final_ori, T, NPoints, grip_cmd):

        x_e_i = np.copy(init_pos)    
        x_e_f = np.copy(final_pos) 
        x_e, dx_e, t = self.plan_pos_trajectory(x_e_i, x_e_f, T, NPoints) 
        
        x_e_i = np.copy(init_ori)      	
        x_e_f = np.copy(final_ori)
        x_e, dx_e, t = self.plan_ori_trajectory(x_e, dx_e, x_e_i, x_e_f, T, NPoints)
        
        g = np.zeros((len(t),3))    
        g[0] = grip_cmd

        return x_e, dx_e, g, t
    
    def path_assembly(self):
        
        size = len(self._points)    
        global_x_e = []  
        global_dx_e = [] 
        global_g = [] 
        global_t = []
        		
        for idx in range(size - 1):		
            p_x_e_i = self._points[idx][0:3]
            p_x_e_f = self._points[idx + 1][0:3]
            trajectory_pos = np.concatenate([p_x_e_i,p_x_e_f])    
            o_x_e_i = self._points[idx][3:6]
            o_x_e_f = self._points[idx + 1][3:6]
            trajectory_ori = np.concatenate([o_x_e_i,o_x_e_f])
            trajectory = np.concatenate([trajectory_pos,trajectory_ori])    
            
            x_e, dx_e, g, t = self.path_planning(trajectory[0:3], trajectory[3:6], trajectory[6:9], trajectory[9:12], self._T[idx], self._NPoints[idx], self._gripper_cmd[idx])
            
            global_x_e.append(x_e)
            global_dx_e.append(dx_e)
            global_g.append(g)
            if idx > 0:
            	global_t.append(t + global_t[idx - 1][-1])
            else:	
            	global_t.append(t)
            
        
        self._x_e = np.concatenate(global_x_e)
        self._dx_e = np.concatenate(global_dx_e)
        self._g = np.concatenate(global_g)
        self._t = np.concatenate(global_t)
    
    def path_assembly_from_arguments(self, points, T, NPoints, gripper_cmd):
        size = len(points)    
        global_x_e = []  
        global_dx_e = [] 
        global_g = [] 
        global_t = []
        		
        for idx in range(size - 1):		
            p_x_e_i = points[idx][0:3]
            p_x_e_f = points[idx + 1][0:3]
            trajectory_pos = np.concatenate([p_x_e_i,p_x_e_f])    
            o_x_e_i = points[idx][3:6]
            o_x_e_f = points[idx + 1][3:6]
            trajectory_ori = np.concatenate([o_x_e_i,o_x_e_f])
            trajectory = np.concatenate([trajectory_pos,trajectory_ori])    
            
            x_e, dx_e, g, t = self.path_planning(trajectory[0:3], trajectory[3:6], trajectory[6:9], trajectory[9:12], T[idx], NPoints[idx], gripper_cmd[idx])
            
            global_x_e.append(x_e)
            global_dx_e.append(dx_e)
            global_g.append(g)
            if idx > 0:
            	global_t.append(t + global_t[idx - 1][-1])
            else:	
            	global_t.append(t)
            
        
        x_e = np.concatenate(global_x_e)
        dx_e = np.concatenate(global_dx_e)
        g = np.concatenate(global_g)
        t = np.concatenate(global_t)
        
        return x_e, dx_e, g, t
            
        
    def target_joints_velocities(self, world, pb):
    
        self._target_joint_angles = np.zeros((len(self._x_e),7))
        
        i = 0        
        while i < len(self._x_e):
              target_whole_joint_angles = world.robot.position_ik(self._x_e[i][0:3], euler_to_quaternion_raw(pb, self._x_e[i][3:6]))  
              self._target_joint_velocities[i] = np.dot(np.linalg.pinv((world.robot.jacobian(target_whole_joint_angles)[:, 0:7])),
                                                    np.array([np.concatenate([self._dx_e[i][0:3], self._dx_e[i][3:6]])]).T).flatten() 
              i+=1                                            
   
    def execute_joints_trajectory_explicitly(self, x_e, dx_e, g, world, slow_rate, pb): 
        
        i = 0        
        while i < len(x_e):
            now = time.time()

            if g[i,[0]] == 1:
                world.robot.gripper_open(pos = g[i,[1]], force = g[i,[2]])                      
            elif g[i,[0]] == -1:
                world.robot.gripper_close(pos = g[i,[1]], force = g[i,[2]])
                          
            goal_pos = np.asarray(x_e[i][0:3]).reshape([3,1])
            goal_ori = np.asarray(euler_to_quaternion_raw(pb, x_e[i][3:6]))           
            goal_vel = np.asarray(dx_e[i][0:3]).reshape([3,1])
            goal_omg = np.asarray(dx_e[i][3:6]).reshape([3,1])           
             
            goal_joint_angles = world.robot.position_ik(goal_pos, goal_ori)
            goal_joint_velocities = np.dot( np.linalg.pinv(world.robot.jacobian(goal_joint_angles)) , np.array([np.concatenate([goal_vel , goal_omg])]).T).flatten()
        
            curr_joint_angles =  world.robot.angles()
            curr_joint_velocities =  world.robot.joint_velocities() 
        
            delta_angles     = goal_joint_angles     - curr_joint_angles
            delta_velocities = goal_joint_velocities - curr_joint_velocities     

            # Desired joint-space torque using PD law
            kP = np.asarray([1000., 1000., 1000., 1000., 1000., 1000., 500., 0., 0.])    
            kD =  np.asarray([2.   , 2.   , 2.  , 2  , 2.  , 2 , 1. , 0., 0.])
            error_thresh = np.asarray([0.010, 0.010])
            # Output controller
            tau = np.add(kP*(delta_angles).T, kD*(delta_velocities).T).reshape(-1)            
            error_angles = np.asarray([np.linalg.norm(delta_angles), np.linalg.norm(delta_velocities)])
        
            # joint torques to be commanded
            torque_cmd = tau + world.robot.torque_compensation()    
        
            world.robot.exec_torque_cmd(torque_cmd)
            world.robot.step_if_not_rtsim()    

            elapsed = time.time() - now
            sleep_time = (1./slow_rate) - elapsed
            if sleep_time > 0.0:
                time.sleep(sleep_time)
            i+=1
     
    def execute_joints_trajectory(self, x_e, dx_e, g, world, controller, slow_rate, pb):  
       
        ee_pos_display = np.zeros((len(x_e),3))
        ee_ori_display = np.zeros((len(x_e),3))
        ee_vel_display = np.zeros((len(x_e),3)) 
        ee_omg_display = np.zeros((len(x_e),3)) 

        joint_angles_display = np.zeros((len(x_e),7))
        joint_velocities_display = np.zeros((len(x_e),7))  
        joint_accelerations_display = np.zeros((len(x_e),7))
        joint_accelerations_filtered_display = np.zeros((len(x_e),7)) 
        total_joint_accelerations_display = np.zeros(len(x_e))  
        target_joint_angles_display = np.zeros((len(x_e),7))
        target_joint_velocities_display = np.zeros((len(x_e),7))
        
        reward = np.zeros(len(x_e))
        
        window_size = 10           
        sampled_values = np.zeros((window_size, 7))

        i = 0   
        avg_idx = 0     
        while i < len(x_e):
            now = time.time()

            ee_pos, ee_ori = world.robot.ee_pose()
            ee_pos_display[i] = ee_pos
            ee_ori_display[i] = quaternion_to_euler_angle(pb, ee_ori.w,ee_ori.x,ee_ori.y,ee_ori.z)
        
            ee_vel, ee_omg = world.robot.ee_velocity()                    
            ee_vel_display[i] = ee_vel
            ee_omg_display[i] = ee_omg
    
            target_whole_joint_angles = world.robot.position_ik(x_e[i][0:3], euler_to_quaternion_raw(pb, x_e[i][3:6]))       
            target_joint_angles_display[i] = target_whole_joint_angles[0:7]       

            target_joint_velocities_display[i] = np.dot(np.linalg.pinv((world.robot.jacobian(target_whole_joint_angles)[:, 0:7])),np.array([np.concatenate([dx_e[i][0:3], dx_e[i][3:6]])]).T).flatten()
            joint_angles_display[i] =  world.robot.angles()[0:7]
            joint_velocities_display[i] =  world.robot.joint_velocities()[0:7]
            joint_accelerations_display[i] = (world.robot.joint_velocities()[0:7]*slow_rate) if i == 0 else ((world.robot.joint_velocities()[0:7] - joint_velocities_display[i-1])*slow_rate) 
            
            if avg_idx > (window_size - 1):
                avg_idx = 0
            sampled_values[avg_idx] = joint_accelerations_display[i]            
            avg_idx +=1            
            avg_values = np.mean(sampled_values, axis=0)          
            
            joint_accelerations_filtered_display[i] = avg_values
            total_joint_accelerations_display[i] = np.linalg.norm(avg_values)
            
            
            X, Y, Z, X_target, Y_target, Z_target = calc_reference_frames(world, self._points[4][0:3], self._points[4][3:6], pb)
        
            delta_x = np.asarray(X_target.reshape(3,1) - X.reshape(3,1))
            err_x = np.linalg.norm(delta_x)            
            delta_y = np.asarray(Y_target.reshape(3,1) - Y.reshape(3,1))
            err_y = np.linalg.norm(delta_y)            
            delta_z = np.asarray(Z_target.reshape(3,1) - Z.reshape(3,1))            
            err_z = np.linalg.norm(delta_z)            
                
            lamba_err = 20.
            lamba_eff = 0.002        
            delta_x_error = err_x + err_y + err_z
            acc_error = np.linalg.norm(total_joint_accelerations_display[i])
            reward[i] = np.exp(-lamba_err*np.square(delta_x_error)) - np.clip(lamba_eff*acc_error, 0, 1)
             
            if g[i,[0]] == 1:
                world.robot.gripper_open(pos = g[i,[1]], force = g[i,[2]])                      
            elif g[i,[0]] == -1:
                world.robot.gripper_close(pos = g[i,[1]], force= g[i,[2]])

            controller.update_goal(x_e[i][0:3], euler_to_quaternion_raw(pb, x_e[i][3:6]), dx_e[i][0:3], dx_e[i][3:6])         

            elapsed = time.time() - now
            sleep_time = (1./slow_rate) - elapsed
            if sleep_time > 0.0:
                time.sleep(sleep_time)
            i+=1           

        sim_outcome = {'ee_pos_display': ee_pos_display, 'ee_ori_display': ee_ori_display, 'ee_vel_display': ee_vel_display, 'ee_omg_display': ee_omg_display, 
                       'joint_angles_display': joint_angles_display, 'joint_velocities_display': joint_velocities_display, 'target_joint_angles_display': target_joint_angles_display, 
                       'target_joint_velocities_display': target_joint_velocities_display, 'joint_accelerations_display': joint_accelerations_display, 'joint_accelerations_filtered_display': joint_accelerations_filtered_display, 'total_joint_accelerations_display': total_joint_accelerations_display, 'reward': reward
                      }
        
        return sim_outcome
        
        
    def execute_task_trajectory(self, x_e, dx_e, g, world, controller, slow_rate, pb):

        
        ee_pos_display = np.zeros((len(x_e),3))
        ee_ori_display = np.zeros((len(x_e),3))
        ee_vel_display = np.zeros((len(x_e),3)) 
        ee_omg_display = np.zeros((len(x_e),3)) 
        
 
        joint_angles_display = np.zeros((len(x_e),7))
        joint_velocities_display = np.zeros((len(x_e),7))  
        joint_accelerations_display = np.zeros((len(x_e),7))
        joint_accelerations_filtered_display = np.zeros((len(x_e),7)) 
        total_joint_accelerations_display = np.zeros(len(x_e))  
        target_joint_angles_display = np.zeros((len(x_e),7))
        target_joint_velocities_display = np.zeros((len(x_e),7))
        
        reward = np.zeros(len(x_e))
        
        window_size = 10           
        sampled_values = np.zeros((window_size, 7))       
    
        i = 0   
        avg_idx = 0        
        while i < len(x_e):
            now = time.time()

            ee_pos, ee_ori = world.robot.ee_pose()
            ee_pos_display[i] = ee_pos
            ee_ori_display[i] = quaternion_to_euler_angle(pb, ee_ori.w,ee_ori.x,ee_ori.y,ee_ori.z)
            
            ee_vel,ee_omg = world.robot.ee_velocity()  
            ee_vel_display[i] = ee_vel
            ee_omg_display[i] = ee_omg 
    
            target_whole_joint_angles = world.robot.position_ik(x_e[i][0:3], euler_to_quaternion_raw(pb, x_e[i][3:6]))       
            target_joint_angles_display[i] = target_whole_joint_angles[0:7]        
            target_joint_velocities_display[i] = np.dot(np.linalg.pinv((world.robot.jacobian(target_whole_joint_angles)[:, 0:7])),np.array([np.concatenate([dx_e[i][0:3], dx_e[i][3:6]])]).T).flatten()
            joint_angles_display[i] =  world.robot.angles()[0:7]
            joint_velocities_display[i] =  world.robot.joint_velocities()[0:7]
            joint_accelerations_display[i] = (world.robot.joint_velocities()[0:7]*slow_rate) if i == 0 else ((world.robot.joint_velocities()[0:7] - joint_velocities_display[i-1])*slow_rate) 
            
            if avg_idx > (window_size - 1):
                avg_idx = 0
            sampled_values[avg_idx] = joint_accelerations_display[i]            
            avg_idx +=1            
            avg_values = np.mean(sampled_values, axis=0)          
            
            joint_accelerations_filtered_display[i] = avg_values
            total_joint_accelerations_display[i] = np.linalg.norm(avg_values)
            
            X, Y, Z, X_target, Y_target, Z_target = calc_reference_frames(world, self._points[4][0:3], self._points[4][3:6], pb)
        
            delta_x = np.asarray(X_target.reshape(3,1) - X.reshape(3,1))
            err_x = np.linalg.norm(delta_x)            
            delta_y = np.asarray(Y_target.reshape(3,1) - Y.reshape(3,1))
            err_y = np.linalg.norm(delta_y)            
            delta_z = np.asarray(Z_target.reshape(3,1) - Z.reshape(3,1))            
            err_z = np.linalg.norm(delta_z)            
                
            lamba_err = 20.
            lamba_eff = 0.002        
            delta_x_error = err_x + err_y + err_z
            acc_error = np.linalg.norm(total_joint_accelerations_display[i])
            reward[i] = np.exp(-lamba_err*np.square(delta_x_error)) - np.clip(lamba_eff*acc_error, 0, 1)
             
            if g[i,[0]] == 1:
                world.robot.gripper_open(pos = g[i,[1]], force = g[i,[2]])                      
            elif g[i,[0]] == -1:
                world.robot.gripper_close(pos = g[i,[1]], force= g[i,[2]])

            controller.update_goal(x_e[i][0:3], euler_to_quaternion_raw(pb, x_e[i][3:6]))    

            elapsed = time.time() - now        
            sleep_time = (1./slow_rate) - elapsed
            if sleep_time > 0.0:
                time.sleep(sleep_time)
            i+=1
            
        sim_outcome = {'ee_pos_display': ee_pos_display, 'ee_ori_display': ee_ori_display, 'ee_vel_display': ee_vel_display, 'ee_omg_display': ee_omg_display, 
                       'joint_angles_display': joint_angles_display, 'joint_velocities_display': joint_velocities_display, 'target_joint_angles_display': target_joint_angles_display, 
                       'target_joint_velocities_display': target_joint_velocities_display, 'joint_accelerations_display': joint_accelerations_display, 'joint_accelerations_filtered_display': joint_accelerations_filtered_display, 'total_joint_accelerations_display': total_joint_accelerations_display, 'reward': reward
                      }
        
        return sim_outcome        
