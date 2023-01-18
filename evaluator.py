import numpy as np
import collections
from datetime import datetime
from visualizer import Plotter
from utils import distance, prepare_data

# prototype evaluator class
# only works for one vehicle evaluation now
# needs to pass in latency term for evaluation
class Evaluator:
    def __init__(self, dis_th, dt, gt, det_freq, gt_freq, latency=None, center_lat=None, center_lon=None, roi_radius=50) -> None:
        self.dis_th = dis_th
        # TODO: compute detection and groundtruth frequency
        # TODO: remove outside data
        self.dtdp_list, self.gtdp_list = prepare_data(dt, gt)
        self.latency = self.compute_latency_method1() if latency is None else latency
        self.det_freq = det_freq
        self.gt_freq = gt_freq
        self.center_lat = center_lat
        self.center_lon = center_lon

    def evaluate(self, visualize=True):
        '''
        Arguments:
            dt: detection results
            gt: ground-truth results
            latency: estimated latency for the perception system
            exp_det_freq: expected detection frequency
        '''
        self._compute_matching()
        if visualize:
            self._visualize_matching()
        self._compute_position_error()
        self._compute_false_positives()
        self._compute_false_negatives()
        self._compute_det_frequency()
        self._compute_lat_lon_error()
        self._compute_id_switch_and_consistency()
        self._compute_mota()
        self._summarize_results()

    def _compute_matching(self):
        dtdp_time = np.array([x.time for x in self.dtdp_list]) - self.latency * 1000000000
        gtdp_time = np.array([x.time for x in self.gtdp_list])
        time_diff = np.abs(dtdp_time[:, None] - gtdp_time[None, :])
        matched_gt_idx = np.argmin(time_diff, axis=1)
        for dp_idx, dtdp in enumerate(self.dtdp_list):
            dtdp.match = matched_gt_idx[dp_idx]
        matched_dt_idx = np.argmin(time_diff, axis=0)
        for dp_idx, gtdp in enumerate(self.gtdp_list):
            gtdp.match = matched_dt_idx[dp_idx]

    def _visualize_matching(self):
        plotter = Plotter(center_lat=self.center_lat, center_lon=self.center_lon)
        plotter.plot_matching(self.dtdp_list, self.gtdp_list)
        plotter.plot_traj_data(self.dtdp_list, plot_name='Msight detection', color='red')
        plotter.plot_traj_data(self.gtdp_list, plot_name='GPS RTK', color='blue')
        plotter.fig.show()

    def _compute_position_error(self):
        '''
        Sign of the errors:
            Lateral error: 
                on the left side of the vehicle heading (-)
                on the right side of the vehicle heading (+)
            Longitudinal error: 
                on the back side of the vehicle heading (-)
                on the front side of the vehicle heading (+)
        '''
        for dtdp in self.dtdp_list:
            if dtdp.match != -1:
                matched_gtdp = self.gtdp_list[dtdp.match]
                gt_heading = matched_gtdp.heading
                # compute distance error between detection and matched ground-truth
                d_lat, d_lon = distance(matched_gtdp.lat, matched_gtdp.lon, dtdp.lat, dtdp.lon)
                d = (d_lat**2 + d_lon**2)**0.5
                # compute heading of the distance error
                d_heading = np.arctan2(d_lon, d_lat)
                # compute the heading offset between distance error and gt heading
                heading_diff = d_heading - gt_heading
                # compute lateral error and longitudinal error w.r.t vehicle heading
                dtdp.lat_error = d * np.sin(heading_diff)
                dtdp.lon_error = d * np.cos(heading_diff)

    def _compute_false_positives(self):
        '''
        FP rate = num_false_positive_detection / num_total_detection
        '''
        for dtdp in self.dtdp_list:
            if dtdp.lat_error is not None or np.abs(dtdp.lat_error) < 1.5:
                dtdp.tp = True

        self.num_false_positive = sum([1.0 for dtdp in self.dtdp_list if dtdp.tp == False])
        self.false_positive_rate = self.num_false_positive / len(self.dtdp_list)
    
    def _compute_false_negatives(self):
        '''
        FN rate = (num_expected_detection - num_true_positive_detection) / num_expected_detection
        '''
        num_exp_det = len(self.gtdp_list) * self.det_freq / float(self.gt_freq)
        num_tp_det =  sum([1.0 for dtdp in self.dtdp_list if dtdp.tp == True])
        self.num_false_negative = max(0.0, num_exp_det - num_tp_det)
        self.false_negative_rate = self.num_false_negative / num_exp_det

    def _compute_det_frequency(self):
        time_interval = (self.dtdp_list[-1].time - self.dtdp_list[0].time) / 1000000000.0
        self.actual_det_freq = (len(self.dtdp_list) - 1) / time_interval

    def _compute_lat_lon_error(self):
        '''
        compute the lateral and longitudinal error for the true positive detections
        '''
        lat_errors = np.abs(np.array([dtdp.lat_error for dtdp in self.dtdp_list if dtdp.tp == True]))
        lon_errors = np.abs(np.array([dtdp.lon_error for dtdp in self.dtdp_list if dtdp.tp == True]))
        self.lat_error = np.mean(lat_errors)
        self.lon_error = np.mean(lon_errors)
    
    def _compute_id_switch_and_consistency(self):
        '''
        compute id switch time for the true positive detections
        '''
        ids = [dtdp.id for dtdp in self.dtdp_list if dtdp.tp == True]
        ids_set = set(ids)
        self.id_switch = len(ids_set) - 1
        id_counts = collections.Counter(ids)
        self.id_consistency = max(id_counts.values()) / len(ids)

    def _compute_mota(self):
        '''
        compute Multiple Object Tracking Accuracy (MOTA) score
        MOTA = 1 - (FN + FP + IDS) / num_expected_detection
        '''
        assert self.false_positive_rate is not None and self.false_negative_rate is not None and self.id_switch is not None, 'need to compute fp, fn, ids first'
        num_exp_det = len(self.gtdp_list) * self.det_freq / float(self.gt_freq)
        self.mota = 1 - (self.num_false_positive + self.num_false_negative + self.id_switch) / num_exp_det

    def _summarize_results(self):
        print('-------------------------------------------------------')
        print('-------------- Evaluation Results ---------------------')
        print('-------------------------------------------------------')
        print(f'False Positive rate\t\t: {self.false_positive_rate:.4f}')
        print(f'False Negative rate\t\t: {self.false_negative_rate:.4f}')
        print(f'Actual detection frequency\t: {self.actual_det_freq:.4f}')
        print(f'Lateral error\t\t\t: {self.lat_error:.4f}')
        print(f'Longitudinal error\t\t: {self.lon_error:.4f}')
        print(f'System Latency\t\t\t: {self.latency:.4f}')
        print(f'ID switch\t\t\t: {self.id_switch:.4f}')
        print(f'ID consistency\t\t\t: {self.id_consistency:.4f}')
        print(f'MOTA\t\t\t\t: {self.mota:.4f}')
        print('-------------------------------------------------------')
        print('---------- End of Evaluation Results ------------------')
        print('-------------------------------------------------------')

    def _compute_minimum_distance_matching(self):
        '''
        compute matching based on minimum distance
        notice: mask out matches with time inverval larger than 2 seonds
        '''
        min_dis_match_idx = []
        for dtdp in self.dtdp_list:
            d_min = 1e8
            match_idx = -1
            gtdp_masked_indices = [idx for (idx, gtdp) in enumerate(self.gtdp_list) if np.abs(gtdp.time - dtdp.time)/1e9 < 2]
            for gtdp_idx in gtdp_masked_indices:
                gtdp = self.gtdp_list[gtdp_idx]
                d_lat, d_lon = distance(dtdp.lat, dtdp.lon, gtdp.lat, gtdp.lon)
                d = (d_lat**2 + d_lon**2)**0.5
                if d < d_min:
                    d_min = d
                    match_idx = gtdp_idx
            min_dis_match_idx.append(match_idx)
        return min_dis_match_idx

    def compute_latency_method1(self):
        '''
        A roundtrip with overlapped trajectories
        First match the detection with the ground-truth points by minimum distance
        Then directly compute the mean time inverval between detection and matched ground-truth as mean latency
        '''
        min_dis_match_idx = self._compute_minimum_distance_matching()
        # plotter = Plotter(center_lat=42.229394, center_lon=-83.739015)
        # plotter.plot_matching(self.dtdp_list, self.gtdp_list, min_dis_match_idx, color='green')
        # plotter.plot_traj_data(self.dtdp_list, plot_name='Msight detection', color='red')
        # plotter.plot_traj_data(self.gtdp_list, plot_name='GPS RTK', color='blue')
        # plotter.fig.show()
        time_diff = np.array([dtdp.time - self.gtdp_list[min_dis_match_idx[i]].time for (i, dtdp) in enumerate(self.dtdp_list)])
        latency = np.mean(time_diff) / 1e9
        return latency



    def compute_latency_method2(self):
        '''
        A roundtrip with overlapped trajectories

        trip 1: from left to right
        trip 2: from right to left

        at a timestamp t, the gt vehicle location is x(t), the detected vehicle location is y(t),
        the detection error is e(t), the system latency is tau(t), vehicle velocity v(t)

        Assume in trip 1, v(t) = v, in trip 2, v(t) = -v

        In trip 1:
        y(t) = x(t - tau(t)) + e(t)
             = x(t) - delta_x(tau(t)) + e(t)
             = x(t) - v * tau(t) + e(t)

        y(t) - x(t) = -v * tau(t) + e(t)
        \sum_t [y(t) - x(t)] = -v \sum_t[tau(t)] + \sum_t[e(t)]
        In trip 2:
        y(t) = x(t) + v * tau(t) + e(t)

        y(t) - x(t) = v * tau(t) + e(t)
        \sum_t [y(t) - x(t)] = v \sum_t[tau(t)] + \sum_t[e(t)]


        '''
        return