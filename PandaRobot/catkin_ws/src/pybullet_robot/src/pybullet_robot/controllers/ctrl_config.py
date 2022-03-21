import numpy as np

KP_P = np.asarray([25000., 25000., 25000.]) # 6000 25000
KP_O = np.asarray([500., 500., 500.]) # 300 700
OSImpConfig = {
    'P_pos': KP_P,
    'D_pos': 2*np.sqrt(KP_P),
    'P_ori': KP_O,
    'D_ori': np.asarray([0.5,0.5,0.5]),
    'kP': np.zeros((9,1)),
    'kD': np.zeros((9,1)),
    'error_thresh': np.asarray([0.005, 0.005]),
    'start_err': np.asarray([200., 200.])
}

OSImpConfigJS = {
    'P_pos': np.zeros((3,1)),
    'D_pos': np.zeros((3,1)),
    'P_ori': np.zeros((3,1)),
    'D_ori': np.zeros((3,1)),     
    'kP': np.asarray([1000., 1000., 1000., 1000., 1000., 1000., 500., 0., 0.]),    
    #'kD': np.asarray([50.   , 50.   , 5.5  , 3.9  , 3.6  , 2.4 , 1.9 , 0., 0.]),
    'kD': np.asarray([   1.,    1.,    1.,    1.,    1.,   0.5,  0.5, 0., 0.]),
    'error_thresh': np.asarray([0.005, 0.005]),
    'start_err': np.asarray([200., 200.])
}

OSHybConfig = {
    'P_pos': KP_P,
    'D_pos': 2*np.sqrt(KP_P),
    'P_ori': KP_O,
    'D_ori': np.asarray([0.01, 0.01, 0.01]),
    'P_f': np.asarray([0.3,0.3,0.3]),
    'P_tor': np.asarray([0.3,0.3,0.3]),
    'I_f': np.asarray([3.,3.,3.]),
    'I_tor': np.asarray([3.,3.,3.]),
    'kP': np.zeros((9,1)),
    'kD': np.zeros((9,1)),
    'error_thresh': np.asarray([0.005, 0.005, 0.5, 0.5]),
    'start_err': np.asarray([200.,200., 200., 200.]),
    'ft_directions': [0,0,0,0,0,0],
    'windup_guard': [100.,100.,100.,100.,100.,100.],
    'null_stiffness': [000.]*7,
}


