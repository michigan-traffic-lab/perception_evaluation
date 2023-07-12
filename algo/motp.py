import numpy as np
from .utils import distance

def compute_motp(dtdp_list):
    '''
        compute the mean error for the true positive detections
    '''
    errors = np.array([dtdp.error for dtdp in dtdp_list if dtdp.tp == True])
    motp = np.mean(errors)
    return motp
            