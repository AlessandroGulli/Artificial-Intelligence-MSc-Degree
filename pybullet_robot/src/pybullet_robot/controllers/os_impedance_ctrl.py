import numpy as np
import quaternion
from pybullet_robot.controllers.utils import quatdiff_in_euler
from pybullet_robot.controllers.os_controller import OSControllerBase
from pybullet_robot.controllers.ctrl_config import OSImpConfig

class OSImpedanceController(OSControllerBase):

    def __init__(self, robot, config=OSImpConfig, **kwargs):

        OSControllerBase.__init__(self, robot=robot, config=config, **kwargs)

    def update_goal(self, goal_pos, goal_ori): 
        self._mutex.acquire()
        self._goal_pos = np.asarray(goal_pos).reshape([3,1])
        self._goal_ori = np.quaternion(goal_ori[3], goal_ori[0], goal_ori[1], goal_ori[2])  
        self._mutex.release()     

    def _compute_cmd(self):
        """
        Actual control loop. Uses goal pose from the feedback thread
        and current robot states from the subscribed messages to compute
        task-space force, and then the corresponding joint torques.
        """
        curr_pos, curr_ori = self._robot.ee_pose()

        delta_pos = self._goal_pos - curr_pos.reshape([3, 1])       
        delta_ori = quatdiff_in_euler(curr_ori, self._goal_ori).reshape([3, 1])   

        curr_vel, curr_omg = self._robot.ee_velocity()    
 
        # print self._goal_pos, curr_pos
        # Desired task-space force using PD law
        F = np.vstack([self._P_pos.dot(delta_pos), self._P_ori.dot(delta_ori)]) - \
            np.vstack([self._D_pos.dot(curr_vel.reshape([3, 1])), self._D_ori.dot(curr_omg.reshape([3, 1]))])  

        error = np.asarray([np.linalg.norm(delta_pos), np.linalg.norm(delta_ori)])

        J = self._robot.jacobian()  
        # joint torques to be commanded
        return np.dot(J.T, F).flatten() + self._robot.torque_compensation(), error        

    def _initialise_goal(self):
        quat_ori = [self._robot.ee_pose()[1].x, self._robot.ee_pose()[1].y, self._robot.ee_pose()[1].z,self._robot.ee_pose()[1].w]        
        self.update_goal(self._robot.ee_pose()[0], quat_ori)



