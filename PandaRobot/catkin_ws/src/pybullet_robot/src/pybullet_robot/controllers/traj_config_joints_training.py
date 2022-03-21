import numpy as np

init_pos = np.asarray([0.31058202         , 0.00566847         , 0.58654213        ])
init_ori = np.asarray([-3.1240766304753516, 0.04029881344985077, 0.0288575068115082])

trajectory = np.array([[init_pos[0], init_pos[1], init_pos[2], init_ori[0], init_ori[1], init_ori[2]],
                       [init_pos[0], init_pos[1], 0.45       , init_ori[0], init_ori[1], init_ori[2]], 
			[init_pos[0], 0.2        , 0.45       , init_ori[0], init_ori[1], init_ori[2]],
			[0.2        , 0.2        , 0.45       , init_ori[0], init_ori[1], init_ori[2]], 
			[init_pos[0], 0.2        , 0.45       , init_ori[0], init_ori[1], init_ori[2]], 
			[init_pos[0], init_pos[1], 0.45       , init_ori[0], init_ori[1], init_ori[2]], 
			[init_pos[0], init_pos[1], init_pos[2], init_ori[0], init_ori[1], init_ori[2]]			
		      ])
T = np.array([1.,1.2,1.2,1.2,2])
grip_cmd = np.array([[-1,0,0],[0,0,0],[1,0.04,0],[0,0,0],[-1,0,0]])
rate = 500. #Hz

Traj_Config = {
'points': trajectory,
'intervals': T,
'NPoints': T*rate,
'gripper_cmd': grip_cmd,
'rate':rate
}





