import numpy as np
from pybullet_robot.controllers.utils import quatdiff_in_euler, euler_to_quaternion_raw, quaternion_to_euler_angle
from pybullet_robot.controllers.os_controller import OSControllerBase
from pybullet_robot.controllers.ctrl_config import OSImpConfig, OSImpConfigJS

class OSImpedanceControllerJointSpace(OSControllerBase):

    def __init__(self, robot, config=OSImpConfigJS, **kwargs):

        OSControllerBase.__init__(self, robot=robot, config=config, **kwargs)

    def update_goal(self, goal_pos, goal_ori, goal_vel , goal_omg): 
        self._mutex.acquire()
        self._goal_pos = np.asarray(goal_pos).reshape([3,1])
        self._goal_ori = np.asarray(goal_ori)           
        self._goal_vel = np.asarray(goal_vel).reshape([3,1])
        self._goal_omg = np.asarray(goal_omg).reshape([3,1])
        self._mutex.release()     

    def _compute_cmd(self):
        """
        Actual control loop. Uses goal pose from the feedback thread
        and current robot states from the subscribed messages to compute
        task-space force, and then the corresponding joint torques.
        """
                
        curr_vel, curr_omg = self._robot.ee_velocity()
        goal_joint_angles = self._robot.position_ik(self._goal_pos, self._goal_ori)
        goal_joint_velocities = np.dot( np.linalg.pinv(self._robot.jacobian(goal_joint_angles)) , np.array([np.concatenate([self._goal_vel , self._goal_omg])]).T).flatten()
        
        curr_joint_angles =  self._robot.angles()
        curr_joint_velocities =  self._robot.joint_velocities() 
        
        delta_angles     = goal_joint_angles     - curr_joint_angles
        delta_velocities = goal_joint_velocities - curr_joint_velocities     
 
        # Desired joint-space torque using PD law
        tau = np.add(self._kP*(delta_angles).T, self._kD*(delta_velocities).T) 

        error = np.asarray([np.linalg.norm(delta_angles), np.linalg.norm(delta_velocities)])

        torque_upper_limits = np.asarray([87, 87, 87, 87, 12, 12, 12, 0, 0])
        torque_lower_limits = torque_upper_limits*(-1)
        
        torque = np.clip(tau + self._robot.torque_compensation(), torque_lower_limits, torque_upper_limits)

        # joint torques to be commanded
        return torque, error       
        

    def _initialise_goal(self):
        quat_ori = [self._robot.ee_pose()[1].x, self._robot.ee_pose()[1].y, self._robot.ee_pose()[1].z,self._robot.ee_pose()[1].w]        
        self.update_goal(self._robot.ee_pose()[0], quat_ori, self._robot.ee_velocity()[0], self._robot.ee_velocity()[1])



