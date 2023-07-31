import numpy as np
import collections
from algo.mota import compute_mota
from algo.motp import compute_motp
from visualizer import Plotter
from utils import dydx_distance, distance
from algo import compute_latency
from algo.utils import DataPoint, TrajectorySet, Trajectory
from algo.hungarian import hungarian_matching
from algo.hota import compute_idf1, compute_hota


# prototype evaluator class
# only works for one vehicle evaluation now
# needs to pass in latency term for evaluation
class Evaluator:
    def __init__(self, cate, dis_th, det_freq, gt_freq, latency=None, center_latlng=None, roi_radius=50,
                 time_th=3) -> None:
        self.cate = cate
        # self.source = source
        self.dis_th = dis_th
        self.time_th = time_th
        self.det_freq = det_freq
        self.gt_freq = gt_freq
        self.center_lat = center_latlng[0]
        self.center_lon = center_latlng[1]
        self.roi_radius = roi_radius
        self.latency = latency

    def evaluate(self, visualize=True):
        '''
        Arguments:
            dt: detection results
            gt: ground-truth results
            latency: estimated latency for the perception system
            exp_det_freq: expected detection frequency
        '''
        return self.false_positive_rate, self.false_negative_rate, self.actual_det_freq, \
            self.lat_error, self.lon_error, self.id_switch, self.id_consistency, self.mota

    def compute_latency(self, dtdps, gtdps):
        self.latency = compute_latency(
            dtdps, gtdps, dis_th=self.dis_th, time_th=self.time_th)
        return self.latency

    def compute_matching_by_time(self, dtdps, gt_traj, inplace=False):
        dtdp_list = dtdps.dp_list
        gtdp_list = gt_traj.dp_list
        if inplace:
            output_dtdp_list = dtdp_list
            output_gtdp_list = gtdp_list
        else:
            output_dtdp_list = dtdp_list.copy()
            output_gtdp_list = gtdp_list.copy()
        dtdp_time = np.array([x.time for x in dtdp_list]
                             ) - self.latency * 1000000000
        gtdp_time = np.array([x.time for x in gtdp_list])
        time_diff = np.abs(dtdp_time[:, None] - gtdp_time[None, :])
        matched_gt_idx = np.argmin(time_diff, axis=1)
        for dp_idx, dtdp in enumerate(output_dtdp_list):
            dtdp.point_wise_match = output_gtdp_list[matched_gt_idx[dp_idx]]
        matched_dt_idx = np.argmin(time_diff, axis=0)
        for dp_idx, gtdp in enumerate(output_gtdp_list):
            gtdp.point_wise_match = output_dtdp_list[matched_dt_idx[dp_idx]]
        return TrajectorySet(output_dtdp_list), Trajectory(output_gtdp_list)
    
    def _match_frames_by_time(self, dtdps, gtdps):
        gt_time = np.array([x for x in gtdps.dataframes.keys()])
        det_time = np.array([x for x in dtdps.dataframes.keys()]) - self.latency * 1000000000
        time_diff = np.abs(det_time[:, None] - gt_time[None, :])
        gt_times = list(gtdps.dataframes.keys())
        det_times = list(dtdps.dataframes.keys())
        # print([x/1000000000 for x in gt_times])
        matched_gt_idx = np.argmin(time_diff, axis=1)
        total_expected_det = 0
        for dp_idx, det_t in enumerate(dtdps.dataframes):
            det_frame = dtdps.dataframes[det_t]
            det_frame.match = gtdps.dataframes[gt_times[matched_gt_idx[dp_idx]]]
            # print(abs(det_time[dp_idx] - gt_times[matched_gt_idx[dp_idx]]) / 1000000000)
            if abs(det_time[dp_idx] - gt_times[matched_gt_idx[dp_idx]]) / 1000000000 > 0.1:
                det_frame.match = None
            if det_frame.match is not None:
                total_expected_det += len(det_frame.match.dp_list)
        matched_dt_idx = np.argmin(time_diff, axis=0)
        for dp_idx, gt_t in enumerate(gtdps.dataframes):
            gt_frame = gtdps.dataframes[gt_t]
            gt_frame.match = dtdps.dataframes[det_times[matched_dt_idx[dp_idx]]]
            # print(abs(gt_t - det_time[matched_dt_idx[dp_idx]]) / 1000000000)
            if abs(gt_t - det_time[matched_dt_idx[dp_idx]]) / 1000000000 > 0.1:
                det_frame.match = None
        return total_expected_det


    # def _compute_matching_multiple_vehicles(self):

    def _visualize_matching(self):
        plotter = Plotter(center_lat=self.center_lat,
                          center_lon=self.center_lon)
        plotter.plot_matching(self.dtdp_list, self.gtdp_list)
        plotter.plot_traj_data(self.dtdp_list, plot_name='Detection')
        plotter.plot_traj_data(
            self.gtdp_list, plot_name='GPS RTK', color='blue')
        plotter.fig.show()

    def compute_and_store_position_error(self, dtdps, gtdps):
        '''
        Sign of the errors:
            Lateral error: 
                on the left side of the vehicle heading (-)
                on the right side of the vehicle heading (+)
            Longitudinal error: 
                on the back side of the vehicle heading (-)
                on the front side of the vehicle heading (+)
        '''
        dtdp_list = dtdps.dp_list
        gtdp_list = gtdps.dp_list
        for dtdp in dtdp_list:
            if dtdp.point_wise_match is not None:
                matched_gtdp = dtdp.point_wise_match
                gt_heading = matched_gtdp.heading
                # compute distance error between detection and matched ground-truth
                d_lat, d_lon = dydx_distance(
                    matched_gtdp.lat, matched_gtdp.lon, dtdp.lat, dtdp.lon)
                d = (d_lat ** 2 + d_lon ** 2) ** 0.5
                # compute heading of the distance error
                d_heading = np.arctan2(d_lon, d_lat)
                # compute the heading offset between distance error and gt heading
                heading_diff = d_heading - gt_heading
                # compute lateral error and longitudinal error w.r.t vehicle heading
                dtdp.lat_error = abs(d * np.sin(heading_diff))
                dtdp.lon_error = abs(d * np.cos(heading_diff))

        return dtdps, gtdps

    def compute_false_positives(self, dtdps):
        '''
        FP rate = num_false_positive_detection / num_total_detection
        '''
        dtdp_list = dtdps.dp_list
        for dtdp in dtdp_list:

            if dtdp.error is not None and dtdp.error < self.dis_th:
                # print('dtdp error', dtdp.error)
                dtdp.tp = True

        num_false_positive = sum([1 for dtdp in dtdp_list if dtdp.tp is False])
        num_true_positive = sum([1 for dtdp in dtdp_list if dtdp.tp is True])
        false_positive_rate = num_false_positive / len(dtdp_list)
        return num_false_positive, false_positive_rate, num_true_positive
    
    def compute_false_negatives(self, dtdps):
        num_fn = 0
        total = 0
        for t, det_frame in dtdps.dataframes.items():
            if det_frame.match is not None:
                gt_frame = det_frame.match
                num_det = len([x for x in det_frame.dp_list if x.tp is True])
                num_fn += len(gt_frame.dp_list) - num_det
                total += len(gt_frame.dp_list)
        return num_fn, num_fn / len(dtdps.dp_list), total
                    

    def number_of_expected_detection(self, gtdps):
        num_exp_det = 0
        # if gtdps is a trajectory
        if isinstance(gtdps, Trajectory):
            gtdp_list = gtdps.dp_list
            num_exp_det += (len(gtdp_list) - 1) * self.det_freq / \
                float(self.gt_freq) + 1
            # print('length of gtdp_list', len(gtdp_list), 'frequency', gtdps.sample_rate, 'detection frequency', self.det_freq, 'num_exp_det', num_exp_det)
            return num_exp_det
        # if gtdps is a trajectory set
        for gtdp_traj in gtdps.trajectories.values():
            gtdp_list = gtdp_traj.dp_list
            num_exp_det += (len(gtdp_list) - 1) * self.det_freq / \
                float(self.gt_freq) + 1
        return num_exp_det

    def compute_false_negatives_by_frequency(self, dtdps, gtdps, num_exp_det=None):
        '''
        FN rate = (num_expected_detection - num_true_positive_detection) / num_expected_detection
        '''
        dtdps_cut, gtdps_cut = self.remove_outside_data(dtdps, gtdps, extra_buffer_for_gt=0)
        dtdp_list = dtdps_cut.dp_list
        # gtdp_list = gtdps_cut.dp_list

        num_tp_det = sum([1.0 for dtdp in dtdp_list if dtdp.tp == True])
        if num_exp_det is None:
            num_exp_det = self.number_of_expected_detection(gtdps_cut)
        # print('expected number of detection', num_exp_det,
        #       'number of detection', num_tp_det)
        num_false_negative = max(0.0, num_exp_det - num_tp_det)
        false_negative_rate = num_false_negative / num_exp_det
        return int(num_false_negative+0.5), false_negative_rate

    # TODO: needs to support multiple vehicles
    # def _compute_det_frequency(self, dtdp_list, gtdp_list):
    #     time_interval = (dtdp_list[-1].time - dtdp_list[0].time) / 1000000000.0
    #     self.actual_det_freq = (len(dtdp_list) - 1) / time_interval
    #     return self.actual_det_freq
    def compute_actual_frequency(self, dtdps):
        times = list(dtdps.dataframes.keys())
        #print(times)
        time_interval = (times[-1] - times[0]) / (len(times) - 1)
        return 1 / (time_interval / 1000000000)
    
    def compute_interval_variance(self, dtdps):
        times = list(dtdps.dataframes.keys())
        time_intervals = np.diff(times)
        return np.var(time_intervals / 1000000000)

    def compute_lat_lon_error(self, dtdps):
        '''
        compute the lateral and longitudinal error for the true positive detections
        '''
        dtdp_list = dtdps.dp_list
        lat_errors = np.abs(
            np.array([dtdp.lat_error for dtdp in dtdp_list if dtdp.tp == True]))
        lon_errors = np.abs(
            np.array([dtdp.lon_error for dtdp in dtdp_list if dtdp.tp == True]))
        lat_error = np.mean(lat_errors)
        lon_error = np.mean(lon_errors)
        return lat_error, lon_error

    def compute_id_switch(self, dtdps, gtdps):
        '''
        compute id switch time for the true positive detections
        '''
        dtdp_list = dtdps.dp_list
        gtdp_list = gtdps.dp_list
        # print(gtdps)
        dt_ids = [dtdp.id for dtdp in dtdp_list if dtdp.tp == True]
        dt_ids_set = set(dt_ids)
        gt_ids = [gtdp.id for gtdp in gtdp_list]
        gt_ids_set = set(gt_ids)
        num_dt_ids = len(dt_ids_set)
        num_gt_ids = len(gt_ids_set)
        # print(
        #     f"number of det ids: {num_dt_ids}, number of gt ids: {num_gt_ids}")
        id_switch = num_dt_ids - num_gt_ids
        # id_counts = collections.Counter(ids)
        # self.id_consistency = max(id_counts.values()) / len(ids)
        return id_switch

    def compute_mota(self, num_false_positive, num_false_negative, id_switch, num_exp_det):
        return compute_mota(num_false_positive, num_false_negative, id_switch, num_exp_det)

    def compute_motp(self, dtdp_list):
        '''
        compute the mean error for the true positive detections
        '''
        return compute_motp(dtdp_list)

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

    def remove_outside_data(self, dtdps, gtdps, radius=None, inplace=False, extra_buffer_for_gt=5):
        if radius is None:
            radius = self.roi_radius
        if inplace:
            output_dtdp_list = dtdps.dp_list
            output_gtdp_list = gtdps.dp_list
        else:
            output_dtdp_list = dtdps.dp_list.copy()
            output_gtdp_list = gtdps.dp_list.copy()

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
            if d >= radius + extra_buffer_for_gt:  # for groundtruth, we want to cover a larger area
                remove_dp_list.append(dp)
        for dp in remove_dp_list:
            output_gtdp_list.remove(dp)

        if not inplace:
            if isinstance(dtdps, TrajectorySet):
                dtdps_out = TrajectorySet(output_dtdp_list)
                gtdps_out = TrajectorySet(output_gtdp_list)
                for t in dtdps_out.trajectories:
                    dtdps_out.trajectories[t].sample_rate = dtdps.trajectories[t].sample_rate
                for t in gtdps_out.trajectories:
                    gtdps_out.trajectories[t].sample_rate = gtdps.trajectories[t].sample_rate
                return dtdps_out, gtdps_out
            elif isinstance(dtdps, Trajectory):
                dtdps_out = Trajectory(output_dtdp_list)
                gtdps_out = Trajectory(output_gtdp_list)
                dtdps_out.sample_rate = dtdps.sample_rate
                gtdps_out.sample_rate = gtdps.sample_rate
                return dtdps_out, gtdps_out

    def clear_match(self, dps):
        for dp in dps.dp_list:
            dp.point_wise_match = None
            dp.traj_wise_match = None
            dp.tp = False

    def match_points(self, dtdps, gtdps):
        def _find_data_point(datapoints, id):
            for dp in datapoints:
                if dp.id == id:
                    return dp
            return None
        self.clear_match(dtdps)
        self.clear_match(gtdps)
        self._match_frames_by_time(dtdps, gtdps)
        for t, det_t in enumerate(dtdps.dataframes):
            det_frame = dtdps.dataframes[det_t]
            if det_frame.match is not None:
                gt_frame = det_frame.match
                match_score = {k1: {k2: 0 for k2 in gt_frame.ids} for k1 in det_frame.ids}
                for dtdp in det_frame.dp_list:
                    for gtdp in gt_frame.dp_list:
                        d = distance(dtdp.lat, dtdp.lon, gtdp.lat, gtdp.lon)
                        match_score[dtdp.id][gtdp.id] = d
                # print(match_score)
                match_result, total_score = hungarian_matching(match_score, minimize=True)
                for dtdp_id in match_result:
                    dtdp = _find_data_point(det_frame.dp_list, dtdp_id)
                    matched_id = match_result[dtdp.id]
                    dtdp.point_wise_match = _find_data_point(gt_frame.dp_list, matched_id)

        
        
        self.compute_and_store_position_error(dtdps, gtdps)
        num_fp, fp_rate, num_tp = self.compute_false_positives(dtdps)
        num_fn, fn_rate, total_expected_detection = self.compute_false_negatives(dtdps)
        # print('num_tp', num_tp, 'num_fp', num_fp, 'fp_rate', fp_rate, 'num_fn', num_fn, 'fn_rate', fn_rate)
        return num_tp, num_fp, num_fn, fp_rate, fn_rate, total_expected_detection


    def match_trajectories(self, dtdps, gtdps, total_expected_detection):
        match_score = {k1: {k2: 0 for k2 in gtdps.ids} for k1 in dtdps.ids}
        for k1, t1 in dtdps.trajectories.items():
            for k2, t2 in gtdps.trajectories.items():
                self.clear_match(t1)
                self.clear_match(t2)
                self.compute_matching_by_time(t1, t2, inplace=True)
                self.compute_and_store_position_error(t1, t2)
                num_fp, fp_rate, num_tp = self.compute_false_positives(t1)
                # print('num_tp', num_tp, 'num_fp', num_fp, 'fp_rate', fp_rate)
                match_score[k1][k2] += num_tp

        match_result, tpa = hungarian_matching(match_score)
        num_total_points = len(dtdps.dp_list)
        fpa = num_total_points - tpa
        fna = total_expected_detection - tpa
        # for k in match_result:
        #     dtdps.trajectories[k].match = gtdps.trajectories[match_result[k]]
        #     t1 = dtdps.trajectories[k]
        #     t2 = gtdps.trajectories[match_result[k]]
        #     self.clear_match(t1)
        #     self.clear_match(t2)
        #     self.compute_matching_by_time(t1, t2, inplace=True)
        #     self.compute_and_store_position_error(t1, t2)
        #     self.compute_false_positives(t1)
        #     fn, fn_rate = self.compute_false_negatives_by_frequency(t1, t2)
        #     fna += fn
        # print('fna', fna)
        return match_result, tpa, fpa, fna
    
    def compute_idf1(self, tpa, fpa, fna):
        return compute_idf1(tpa, fpa, fna)
    
    def compute_hota(self, tpa, fpa, fna, tp, fp, fn):
        return compute_hota(tpa, fpa, fna, tp, fp, fn)
