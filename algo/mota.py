def compute_mota(num_fn, num_fp, num_ids, num_exp_det):
    '''
    compute Multiple Object Tracking Accuracy (MOTA) score
    MOTA = 1 - (FN + FP + IDS) / num_expected_detection
    '''
    mota = 1 - (num_fp + num_fn + num_ids) / num_exp_det
    return mota
