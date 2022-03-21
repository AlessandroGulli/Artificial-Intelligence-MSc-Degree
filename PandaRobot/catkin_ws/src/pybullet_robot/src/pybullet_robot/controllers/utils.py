import quaternion
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d
import time
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, freqz



def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


def calc_reference_frames(world, target_pos, target_ori, pb):

    curr_pos, curr_ori = world.robot.ee_pose()  
    r = np.asarray(pb.getMatrixFromQuaternion([curr_ori.x, curr_ori.y, curr_ori.z,curr_ori.w]) ).reshape(3,3)
    euler = pb.getEulerFromQuaternion([curr_ori.x, curr_ori.y, curr_ori.z,curr_ori.w])
        
    target_ori = euler_to_quaternion_raw(pb, target_ori)
    r_target = np.asarray(pb.getMatrixFromQuaternion([target_ori[0], target_ori[1], target_ori[2],target_ori[3]]) ).reshape(3,3)    
    euler_target = pb.getEulerFromQuaternion([target_ori[0], target_ori[1], target_ori[2],target_ori[3]])
    
    dx = np.array([0.3, 0. , 0. ])
    dy = np.array([0. , 0.3, 0. ])
    dz = np.array([0. , 0. , 0.3])         

    X = np.add(curr_pos, np.dot(dx.T,r))
    Y = np.add(curr_pos, np.dot(dy.T,r))
    Z = np.add(curr_pos, np.dot(dz.T,r))
    
    X_target = np.add(target_pos, np.dot(dx.T,r_target))
    Y_target = np.add(target_pos, np.dot(dy.T,r_target))
    Z_target = np.add(target_pos, np.dot(dz.T,r_target))
    
    return X, Y, Z, X_target, Y_target, Z_target

        
def render_reference_frames(world, target_pos, target_ori, pb):
             
    red    = [0.97, 0.25, 0.25, 1]
    green  = [0.41, 0.68, 0.31, 1]
    yellow = [0.92, 0.73, 0, 1]
    blue   = [0, 0.55, 0.81, 1] 

    X, Y, Z, X_target, Y_target, Z_target = calc_reference_frames(world, target_pos, target_ori, pb)
    
    pb.addUserDebugLine(world.robot.ee_pose()[0], Z, lineColorRGB=blue[0:3], lineWidth=2.0, lifeTime=0)
    pb.addUserDebugLine(world.robot.ee_pose()[0], Y, lineColorRGB=green[0:3], lineWidth=2.0, lifeTime=0)
    pb.addUserDebugLine(world.robot.ee_pose()[0], X, lineColorRGB=red[0:3], lineWidth=2.0, lifeTime=0)

    pb.addUserDebugLine(world.robot.ee_pose()[0], Z_target, lineColorRGB=yellow[0:3], lineWidth=2.0, lifeTime=0)
    pb.addUserDebugLine(world.robot.ee_pose()[0], Y_target, lineColorRGB=yellow[0:3], lineWidth=2.0, lifeTime=0)
    pb.addUserDebugLine(world.robot.ee_pose()[0], X_target, lineColorRGB=yellow[0:3], lineWidth=2.0, lifeTime=0)


def sample_torus_coordinates(r, R, rot, n):

    radius = np.random.uniform(0, r, n)
    phi = np.random.uniform(-rot, rot, n)
    theta = np.random.uniform(0, 2*np.pi, n)

    x = (R + radius * np.cos(theta)) * np.cos(phi)
    y = (R + radius * np.cos(theta)) * np.sin(phi)
    z = radius * np.sin(theta) + 0.5

    out = np.array([x, y, z], dtype=np.float32)
    return out


def weighted_minkowskian_distance(diff_pos, diff_ori, w_rot=1.5, p=2):

    abs_pos = np.abs(diff_pos)
    abs_ori = w_rot*np.abs(diff_ori)
    dist = np.power(np.sum(np.power(abs_pos,p)) + np.sum(np.power(abs_ori,p)),1/p)    
    return dist

def quaternion_to_euler_angle(pb, w, x, y, z):    
    euler_ang = pb.getEulerFromQuaternion([x, y, z, w])
    return euler_ang
    
def euler_to_quaternion(pb, euler):
    q = pb.getQuaternionFromEuler(euler)         
    return np.quaternion(q[0], q[1], q[2], q[3])
   
def euler_to_quaternion_raw(pb, euler):
    q = pb.getQuaternionFromEuler(euler) 
         
    return q    

def quatdiff_in_euler(quat_curr, quat_des):

    curr_mat = quaternion.as_rotation_matrix(quat_curr)
    des_mat = quaternion.as_rotation_matrix(quat_des)

    rel_mat = des_mat.T.dot(curr_mat)

    rel_quat = quaternion.from_rotation_matrix(rel_mat)

    vec = quaternion.as_float_array(rel_quat)[1:]

    if rel_quat.w < 0.0:
        vec = -vec

    return -des_mat.dot(vec)

def display_trajectories(sim_outcome, x_e, dx_e, t):

    fig1 = plt.figure(figsize=(20,10), constrained_layout=True)
    ax1 = fig1.add_subplot(221)
    ax2 = fig1.add_subplot(222)
    ax3 = fig1.add_subplot(223)
    ax4 = fig1.add_subplot(224)
 
    ax1.set_title("EE Position vs Target Position")  
    ax1.plot(t, sim_outcome['ee_pos_display'][:,[0]], 'r-', label ='EE_Pos X')
    ax1.plot(t, sim_outcome['ee_pos_display'][:,[1]], 'g-' , label ='EE_Pos Y')
    ax1.plot(t, sim_outcome['ee_pos_display'][:,[2]], 'b-' , label ='EE_Pos Z')
    ax1.plot(t, x_e[:,[0]], 'r--', label ='Target_Pos X')
    ax1.plot(t, x_e[:,[1]], 'g--', label ='Target_Pos Y')
    ax1.plot(t, x_e[:,[2]], 'b--', label ='Target_Pos Z') 
    ax1.set_xlabel("time [s]")   
    ax1.set_ylabel("xyz axis [m]") 
    ax1.legend() 
    
    ax2.set_title("EE Orientation vs Target Orientation")  
    ax2.plot(t, sim_outcome['ee_ori_display'][:,[0]], 'r-', label ='EE_Ori X')
    ax2.plot(t, sim_outcome['ee_ori_display'][:,[1]], 'g-', label ='EE_Ori Y')
    ax2.plot(t, sim_outcome['ee_ori_display'][:,[2]], 'b-', label ='EE_Ori Z')
    ax2.plot(t, x_e[:,[3]], 'r--', label ='Target_Ori X')
    ax2.plot(t, x_e[:,[4]], 'g--', label ='Target_Ori Y')
    ax2.plot(t, x_e[:,[5]], 'b--', label ='Target_Ori Z') 
    ax2.set_xlabel("time [s]")   
    ax2.set_ylabel("theta phi psi angles [rad]")
    ax2.legend() 
    
    ax3.set_title("EE Linear Velocity vs Target Linear Velocity")
    ax3.plot(t, sim_outcome['ee_vel_display'][:,[0]],  'r-', label ='EE_Vel X')  
    ax3.plot(t, sim_outcome['ee_vel_display'][:,[1]],  'g-', label ='EE_Vel Y')  
    ax3.plot(t, sim_outcome['ee_vel_display'][:,[2]],  'b-', label ='EE_Vel Z')
    ax3.plot(t, dx_e[:,[0]], 'r--', label ='Target_Vel X')
    ax3.plot(t, dx_e[:,[1]], 'g--', label ='Target_Vel Y')
    ax3.plot(t, dx_e[:,[2]], 'b--', label ='Target_Vel Z')
    ax3.set_xlabel("time [s]")   
    ax3.set_ylabel("dot xyz axis [m/s]") 
    ax3.legend() 
 
    ax4.set_title("EE Angular Velocity vs Target Angular Velocity")  
    ax4.plot(t, sim_outcome['ee_omg_display'][:,[0]], 'r-', label ='EE_Omg X')
    ax4.plot(t, sim_outcome['ee_omg_display'][:,[1]], 'g-', label ='EE_Omg Y')
    ax4.plot(t, sim_outcome['ee_omg_display'][:,[2]], 'b-', label ='EE_Omg Z')
    ax4.plot(t, dx_e[:,[3]], 'r--', label ='Target_Omg X')
    ax4.plot(t, dx_e[:,[4]], 'g--', label ='Target_Omg Y')
    ax4.plot(t, dx_e[:,[5]], 'b--', label ='Target_Omg Z') 
    ax4.set_xlabel("time [s]")   
    ax4.set_ylabel("dot theta phi psi angles [rad/s]")
    ax4.legend()  

    fig2 = plt.figure(figsize=(20,10), constrained_layout=True)    
    ax5 = fig2.add_subplot(211)
    ax6 = fig2.add_subplot(212)  
    
    ax5.set_title("Joint Angles")    
    ax5.plot(t, sim_outcome['joint_angles_display'][:,[0]], 'r-', label ='J0')
    ax5.plot(t, sim_outcome['joint_angles_display'][:,[1]], 'g-', label ='J1')
    ax5.plot(t, sim_outcome['joint_angles_display'][:,[2]], 'b-', label ='J2')
    ax5.plot(t, sim_outcome['joint_angles_display'][:,[3]], 'c-', label ='J3')
    ax5.plot(t, sim_outcome['joint_angles_display'][:,[4]], 'm-', label ='J4')
    ax5.plot(t, sim_outcome['joint_angles_display'][:,[5]], 'y-', label ='J5') 
    ax5.plot(t, sim_outcome['joint_angles_display'][:,[6]], 'k-', label ='J6') 
    ax5.plot(t, sim_outcome['target_joint_angles_display'][:,[0]], 'r--', label ='Target J0')
    ax5.plot(t, sim_outcome['target_joint_angles_display'][:,[1]], 'g--', label ='Target J1')
    ax5.plot(t, sim_outcome['target_joint_angles_display'][:,[2]], 'b--', label ='Target J2')
    ax5.plot(t, sim_outcome['target_joint_angles_display'][:,[3]], 'c--', label ='Target J3')
    ax5.plot(t, sim_outcome['target_joint_angles_display'][:,[4]], 'm--', label ='Target J4')
    ax5.plot(t, sim_outcome['target_joint_angles_display'][:,[5]], 'y--', label ='Target J5')
    ax5.plot(t, sim_outcome['target_joint_angles_display'][:,[6]], 'k--', label ='Target J6') 
    ax5.set_xlabel("time [s]")   
    ax5.set_ylabel("q(t) [rad]") 
    ax5.legend()  
   
    ax6.set_title("Joint Velocities")  
    ax6.plot(t, sim_outcome['joint_velocities_display'][:,[0]], 'r-', label ='J0')
    ax6.plot(t, sim_outcome['joint_velocities_display'][:,[1]], 'g-', label ='J1')
    ax6.plot(t, sim_outcome['joint_velocities_display'][:,[2]], 'b-', label ='J2')
    ax6.plot(t, sim_outcome['joint_velocities_display'][:,[3]], 'c-', label ='J3')
    ax6.plot(t, sim_outcome['joint_velocities_display'][:,[4]], 'm-', label ='J4')
    ax6.plot(t, sim_outcome['joint_velocities_display'][:,[5]], 'y-', label ='J5') 
    ax6.plot(t, sim_outcome['joint_velocities_display'][:,[6]], 'k-', label ='J6') 
    ax6.plot(t, sim_outcome['target_joint_velocities_display'][:,[0]], 'r--', label ='Target J0')
    ax6.plot(t, sim_outcome['target_joint_velocities_display'][:,[1]], 'g--', label ='Target J1')
    ax6.plot(t, sim_outcome['target_joint_velocities_display'][:,[2]], 'b--', label ='Target J2')
    ax6.plot(t, sim_outcome['target_joint_velocities_display'][:,[3]], 'c--', label ='Target J3')
    ax6.plot(t, sim_outcome['target_joint_velocities_display'][:,[4]], 'm--', label ='Target J4')
    ax6.plot(t, sim_outcome['target_joint_velocities_display'][:,[5]], 'y--', label ='Target J5') 
    ax6.plot(t, sim_outcome['target_joint_velocities_display'][:,[6]], 'k--', label ='Target J6') 
    ax6.set_xlabel("time [s]")   
    ax6.set_ylabel("dq(t) [rad/s]")
    ax6.legend()  
    
    fig3 = plt.figure(figsize=(20,10), constrained_layout=True)    
    ax7 = fig3.add_subplot(211)
    ax8 = fig3.add_subplot(212)    
    
    ax7.set_title("Joint Accelerations")  
    ax7.plot(t, sim_outcome['joint_accelerations_display'][:,[0]], 'r-', label ='J0')
    ax7.plot(t, sim_outcome['joint_accelerations_display'][:,[1]], 'g-', label ='J1')
    ax7.plot(t, sim_outcome['joint_accelerations_display'][:,[2]], 'b-', label ='J2')
    ax7.plot(t, sim_outcome['joint_accelerations_display'][:,[3]], 'c-', label ='J3')
    ax7.plot(t, sim_outcome['joint_accelerations_display'][:,[4]], 'm-', label ='J4')
    ax7.plot(t, sim_outcome['joint_accelerations_display'][:,[5]], 'y-', label ='J5') 
    ax7.plot(t, sim_outcome['joint_accelerations_display'][:,[6]], 'k-', label ='J6') 
    ax7.set_xlabel("time [s]")   
    ax7.set_ylabel("ddq(t) [rad/s^2]")
    ax7.legend()
    
    ax8.set_title("Joint Accelerations Filtered")  
    ax8.plot(t, sim_outcome['joint_accelerations_filtered_display'][:,[0]], 'r-', label ='J0')
    ax8.plot(t, sim_outcome['joint_accelerations_filtered_display'][:,[1]], 'g-', label ='J1')
    ax8.plot(t, sim_outcome['joint_accelerations_filtered_display'][:,[2]], 'b-', label ='J2')
    ax8.plot(t, sim_outcome['joint_accelerations_filtered_display'][:,[3]], 'c-', label ='J3')
    ax8.plot(t, sim_outcome['joint_accelerations_filtered_display'][:,[4]], 'm-', label ='J4')
    ax8.plot(t, sim_outcome['joint_accelerations_filtered_display'][:,[5]], 'y-', label ='J5') 
    ax8.plot(t, sim_outcome['joint_accelerations_filtered_display'][:,[6]], 'k-', label ='J6')
    ax8.set_xlabel("time [s]")   
    ax8.set_ylabel("ddq(t) [rad/s^2]")
    ax8.legend() 
    
    fig4 = plt.figure(figsize=(20,10), constrained_layout=True)    
    ax9 = fig4.add_subplot(211)
    ax10 = fig4.add_subplot(212)
    
    ax9.set_title("Total Joints Acceleration")  
    ax9.plot(t, sim_outcome['total_joint_accelerations_display'], 'r-')
    ax9.set_xlabel("time [s]")   
    ax9.set_ylabel("ddq(t) [rad/s^2]")
    ax9.legend()  
    
    ax10.set_title("Reward")  
    ax10.plot(t, sim_outcome['reward'], 'c-')
    ax10.set_xlabel("time [s]")   
    ax10.legend()       
 
    plt.show()
    
