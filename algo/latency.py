import numpy as np
from .utils import compute_minimum_distance_matching
def compute_latency(dtdp_list, gtdp_list, dis_th=1.5, time_th=3):
    '''
    A roundtrip with overlapped trajectories
    First match the detection with the ground-truth points by minimum distance
    Then directly compute the mean time inverval between detection and matched ground-truth as mean latency
    '''
    # dtdp_list_for_latency, gtdp_list_for_latency = self._remove_outside_data(
    #     inplace=False, radius=20)
    min_dis_match_idx = compute_minimum_distance_matching(
        dtdp_list, gtdp_list, dis_th=dis_th, time_th=time_th)
    
    print(f'there are {min_dis_match_idx.count(-1)} unmatched detections')
    #print(dtdp_list)
    #print(gtdp_list)
    # print(min_dis_match_idx)
    # plotter = Plotter(center_lat=self.center_lat, center_lon=self.center_lon)
    # plotter.plot_matching(dtdp_list_for_latency, gtdp_list_for_latency, min_dis_match_idx, color='green')
    # plotter.plot_traj_data(dtdp_list_for_latency, plot_name='Msight detection', color='red')
    # plotter.plot_traj_data(gtdp_list_for_latency, plot_name='GPS RTK', color='blue')
    # plotter.fig.show()
    time_diff = np.array([dtdp.time - gtdp_list[min_dis_match_idx[i]].time for (i, dtdp) in
                          enumerate(dtdp_list) if min_dis_match_idx[i] != -1])
    latency = np.mean(time_diff) / 1e9
    return latency


# def compute_latency_latency_from_experiment(self):

#     return
