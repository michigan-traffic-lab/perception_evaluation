import numpy as np
from geopy import distance as geodistance


def compute_minimum_distance_matching(dtdp_list, gtdp_list, dis_th=1.5, time_th=3):
    '''
    compute matching based on minimum distance
    notice: mask out matches with time inverval larger than 2 seonds
    '''
    min_dis_match_idx = []
    for dtdp in dtdp_list:
        d_min = dis_th
        match_idx = -1
        masked_gtdp_with_idx = [(idx, gtdp) for (idx, gtdp) in enumerate(gtdp_list) if
                                np.abs(gtdp.time - dtdp.time) / 1e9 < time_th]
        for gtdp_idx, gtdp in masked_gtdp_with_idx:
            d = distance(dtdp.lat, dtdp.lon, gtdp.lat, gtdp.lon)
            if d < d_min:
                d_min = d
                match_idx = gtdp_idx
        min_dis_match_idx.append(match_idx)
    return min_dis_match_idx

def distance(lat1, lon1, lat2, lon2):
    return geodistance.geodesic([lat2, lon2], [lat1, lon1]).m
