import numpy as np
import collections
from visualizer import Plotter
from utils import dydx_distance, prepare_data, distance


# prototype evaluator class
# only works for one vehicle evaluation now
# needs to pass in latency term for evaluation
class Evaluator:
    def __init__(self, source, cate, dis_th, dt, gt, det_freq, gt_freq, latency=None, center_latlng=None, roi_radius=50,
                 time_th=3) -> None:
        self.cate = cate
        self.dis_th = dis_th
        self.time_th = time_th
        self.det_freq = det_freq
        self.gt_freq = gt_freq
        self.center_lat = center_latlng[0]
        self.center_lon = center_latlng[1]
        self.roi_radius = roi_radius
        self.dtdp_list, self.gtdp_list = prepare_data(dt, gt, cate=cate, source=source)
        self.latency = self.compute_latency_method1() if latency is None else latency
        self._compute_det_frequency()
        self._remove_outside_data()

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
            self.visualize_data()
        self._compute_position_error()
        self._compute_false_positives()
        self._compute_false_negatives()
        self._compute_lat_lon_error()
        self._compute_id_switch_and_consistency()
        self._compute_mota()
        self._summarize_results()
        return self.false_positive_rate, self.false_negative_rate, self.actual_det_freq, \
            self.lat_error, self.lon_error, self.id_switch, self.id_consistency, self.mota

    def compute_latency(self):
        return self.latency

    def visualize_data(self):
        plotter = Plotter(center_lat=self.center_lat, center_lon=self.center_lon)
        plotter.plot_traj_data(self.gtdp_list, plot_name='GPS RTK', color='blue')
        plotter.plot_traj_data(self.dtdp_list, plot_name='Detection')
        plotter.fig.show()

    def visualize_data_temp(self, dtdp_list, gtdp_lists):
        dtdp_list, gtdp_lists[0] = self._remove_outside_data(False, dtdp_list, gtdp_lists[0], radius=50)
        dtdp_list, gtdp_lists[1] = self._remove_outside_data(False, dtdp_list, gtdp_lists[1], radius=50)
        plotter = Plotter(center_lat=self.center_lat, center_lon=self.center_lon)
        plotter.plot_traj_data(gtdp_lists[0], plot_name='GPS RTK 1', color='blue')
        plotter.plot_traj_data(gtdp_lists[1], plot_name='GPS RTK 2', color='red')
        plotter.plot_traj_data(dtdp_list, plot_name='Detection')
        plotter.fig.show()

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

    # def _compute_matching_multiple_vehicles(self):

    def _visualize_matching(self):
        plotter = Plotter(center_lat=self.center_lat, center_lon=self.center_lon)
        plotter.plot_matching(self.dtdp_list, self.gtdp_list)
        plotter.plot_traj_data(self.dtdp_list, plot_name='Detection')
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
                d_lat, d_lon = dydx_distance(matched_gtdp.lat, matched_gtdp.lon, dtdp.lat, dtdp.lon)
                d = (d_lat ** 2 + d_lon ** 2) ** 0.5
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
            if dtdp.lat_error is not None and np.abs(dtdp.lat_error) < self.dis_th:
                dtdp.tp = True

        self.num_false_positive = sum([1.0 for dtdp in self.dtdp_list if dtdp.tp is False])
        self.false_positive_rate = self.num_false_positive / len(self.dtdp_list)

    # TODO: needs more discussion and rewrite
    def _compute_false_negatives(self):
        '''
        FN rate = (num_expected_detection - num_true_positive_detection) / num_expected_detection
        '''
        num_exp_det = len(self.gtdp_list) * self.det_freq / float(self.gt_freq)
        num_tp_det = sum([1.0 for dtdp in self.dtdp_list if dtdp.tp == True])
        self.num_false_negative = max(0.0, num_exp_det - num_tp_det)
        self.false_negative_rate = self.num_false_negative / num_exp_det

    # TODO: needs to support multiple vehicles
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
        print('-------------- Evaluation Results ---------------------')
        print(f'False Positive rate\t\t: {self.false_positive_rate:.4f}')
        print(f'False Negative rate\t\t: {self.false_negative_rate:.4f}')
        print(f'Actual detection frequency\t: {self.actual_det_freq:.4f}')
        print(f'Lateral error\t\t\t: {self.lat_error:.4f}')
        print(f'Longitudinal error\t\t: {self.lon_error:.4f}')
        print(f'System Latency\t\t\t: {self.latency:.4f}')
        print(f'ID switch\t\t\t: {self.id_switch:.4f}')
        print(f'ID consistency\t\t\t: {self.id_consistency:.4f}')
        print(f'MOTA\t\t\t\t: {self.mota:.4f}')

    def _compute_minimum_distance_matching(self, dtdp_list, gtdp_list):
        '''
        compute matching based on minimum distance
        notice: mask out matches with time inverval larger than 2 seonds
        '''
        min_dis_match_idx = []
        for dtdp in dtdp_list:
            d_min = self.dis_th
            match_idx = -1
            masked_gtdp_with_idx = [(idx, gtdp) for (idx, gtdp) in enumerate(gtdp_list) if
                                   np.abs(gtdp.time - dtdp.time) / 1e9 < self.time_th]
            for gtdp_idx, gtdp in masked_gtdp_with_idx:
                d = distance(dtdp.lat, dtdp.lon, gtdp.lat, gtdp.lon)
                if d < d_min:
                    d_min = d
                    match_idx = gtdp_idx
            min_dis_match_idx.append(match_idx)
        return min_dis_match_idx

    def _remove_outside_data(self, inplace=True, dtdp_list=None, gtdp_list=None, radius=None):
        if radius is None:
            radius = self.roi_radius
        if inplace:
            output_dtdp_list = self.dtdp_list
            output_gtdp_list = self.gtdp_list
        elif dtdp_list is None and gtdp_list is None:
            output_dtdp_list = self.dtdp_list.copy()
            output_gtdp_list = self.gtdp_list.copy()
        else:
            output_dtdp_list = dtdp_list.copy()
            output_gtdp_list = gtdp_list.copy()

        remove_dp_list = []
        for dp in output_dtdp_list:
            d = distance(self.center_lat, self.center_lon, dp.lat, dp.lon)
            if d >= radius:
                remove_dp_list.append(dp)
        for dp in remove_dp_list:
            output_dtdp_list.remove(dp)

        remove_dp_list = []
        for dp in output_gtdp_list:
            d = distance(self.center_lat, self.center_lon, dp.lat, dp.lon)
            if d >= radius:
                remove_dp_list.append(dp)
        for dp in remove_dp_list:
            output_gtdp_list.remove(dp)

        if not inplace:
            return output_dtdp_list, output_gtdp_list

    def compute_latency_method1(self):
        '''
        A roundtrip with overlapped trajectories
        First match the detection with the ground-truth points by minimum distance
        Then directly compute the mean time inverval between detection and matched ground-truth as mean latency
        '''
        dtdp_list_for_latency, gtdp_list_for_latency = self._remove_outside_data(inplace=False, radius=20)
        min_dis_match_idx = self._compute_minimum_distance_matching(dtdp_list_for_latency, gtdp_list_for_latency)
        # plotter = Plotter(center_lat=self.center_lat, center_lon=self.center_lon)
        # plotter.plot_matching(dtdp_list_for_latency, gtdp_list_for_latency, min_dis_match_idx, color='green')
        # plotter.plot_traj_data(dtdp_list_for_latency, plot_name='Msight detection', color='red')
        # plotter.plot_traj_data(gtdp_list_for_latency, plot_name='GPS RTK', color='blue')
        # plotter.fig.show()
        time_diff = np.array([dtdp.time - gtdp_list_for_latency[min_dis_match_idx[i]].time for (i, dtdp) in
                              enumerate(dtdp_list_for_latency) if min_dis_match_idx[i] != -1])
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
