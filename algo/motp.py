import numpy as np
# from .utils import distance

def compute_motp(dtdps):
    '''
        compute the mean error for the true positive detections
    '''
    dtdp_list = dtdps.dp_list
    errors = np.array([dtdp.error for dtdp in dtdp_list if dtdp.tp == True])
    motp = np.mean(errors)
    return motp
