import numpy as np
from geopy import distance as geodistance
from algo.utils import DataPoint, TrajectorySet, DataFrame
import json
from tqdm import tqdm
from tabulate import tabulate


def read_data(data_path, kind='all'):
    with open(data_path, 'r') as f:
        data = json.load(f)
    dp_list = []
    dataframes = {}
    for d in tqdm(data, desc="Loading data", unit="timestamp"):
        t = int(d['timestamp'] * 1e9)
        df = DataFrame([], time=t)
        for o in d['objs']:
            if kind == 'veh':
                if o['classType'] != '2':
                    continue
            elif kind == 'ped':
                if o['classType'] != '10':
                    continue
            dp = DataPoint(o['id'], t, o['lat'], o['lon'])
            df.add_dp(dp)
            dp_list.append(dp)
        dataframes[t] = df

    return TrajectorySet(dp_list)


def dydx_distance(lat1, lon1, lat2, lon2):
    # dx, dy = (lat2 - lat1) * 111000., (lon2 - lon1) * 111000. * np.cos(lat2/180*np.pi)
    dx = geodistance.geodesic(
        [lat2, lon1], [lat1, lon1]).m * np.sign(lat2 - lat1)
    dy = geodistance.geodesic(
        [lat2, lon2], [lat2, lon1]).m * np.sign(lon2 - lon1)
    return dx, dy


def distance(lat1, lon1, lat2, lon2):
    return geodistance.geodesic([lat2, lon2], [lat1, lon1]).m


def compute_speed_and_heading(dp_list, interval=2):
    '''
    interval in each direction (past, future)
    '''
    for i in range(len(dp_list)):
        dp = dp_list[i]
        past_t = max(0, i - interval)
        future_t = min(len(dp_list) - 1, i + interval)
        dy, dx = dydx_distance(
            dp_list[past_t].lat, dp_list[past_t].lon, dp_list[future_t].lat, dp_list[future_t].lon)

        # compute speed m/s
        dp.speed = (dx ** 2 + dy ** 2) ** 0.5 / \
            (dp_list[future_t].time - dp_list[past_t].time) * 1000000000

        # compute heading
        # clock-wise, in degree
        # north: 0, east: pi/2, north-east: pi/4
        dp.heading = np.arctan2(dx, dy)

# TODO: parse in a function to compute speed_heading


def prepare_data(dt, gt, cate='veh', source='Bluecity'):
    # convert detections to datapoints
    dtdp_list = []
    for i in range(len(dt.lats)):
        if source == 'Bluecity':
            if cate == 'veh':
                # for bluecity, veh is type=2
                if dt.obj_type[i] == '2':
                    dtdp_list.append(
                        DataPoint(id=dt.obj_ids[i],
                                  time=float(dt.datetime[i]),
                                  lat=dt.lats[i],
                                  lon=dt.longs[i],
                                  speed=dt.speeds[i]
                                  )
                    )
            elif cate == 'ped':
                # for bluecity, ped is type=2
                if dt.obj_type[i] == '10':
                    dtdp_list.append(
                        DataPoint(id=dt.obj_ids[i],
                                  time=float(dt.datetime[i]),
                                  lat=dt.lats[i],
                                  lon=dt.longs[i],
                                  speed=dt.speeds[i]
                                  )
                    )
            elif cate == 'all':
                dtdp_list.append(
                    DataPoint(id=dt.obj_ids[i],
                              time=float(dt.datetime[i]),
                              lat=dt.lats[i],
                              lon=dt.longs[i],
                              speed=dt.speeds[i]
                              )
                )
        if source == 'Derq':
            dtdp_list.append(
                DataPoint(id=dt.obj_ids[i],
                          time=float(dt.datetime[i]),
                          lat=dt.lats[i],
                          lon=dt.longs[i],
                          speed=dt.speeds[i]
                          )
            )
        if source == 'MSight':
            dtdp_list.append(
                DataPoint(id=dt.obj_ids[i],
                          time=float(dt.datetime[i]),
                          lat=dt.lats[i],
                          lon=dt.longs[i],
                          speed=dt.speeds[i]
                          )
            )

    dtdp_list = sorted(dtdp_list, key=lambda x: x.time)

    # convert groundtruths to datapoints
    gtdp_list = []
    for i in range(len(gt.lats)):
        gtdp_list.append(
            DataPoint(id=0,
                      time=float(gt.datetime[i]),
                      lat=gt.lats[i],
                      lon=gt.longs[i],
                      speed=None
                      )
        )
    gtdp_list = sorted(gtdp_list, key=lambda x: x.time)
    compute_speed_and_heading(dtdp_list, interval=2)
    compute_speed_and_heading(gtdp_list, interval=10)

    return TrajectorySet(dtdp_list), TrajectorySet(gtdp_list)


def get_detection_file_path(system, data_dir, trial_id, vehicle_id=None, object=None):
    if system == 'Bluecity':
        if object == "ped":
            return f'{data_dir}/dets/detection_ped_{trial_id}.json'
        if vehicle_id == None:
            return f'{data_dir}/dets/detection_{trial_id}.json'
        else:
            return f'{data_dir}/dets/detection_{trial_id}_{vehicle_id}.json'
    elif system == 'Derq':
        if object == "ped":
            return f'{data_dir}/dets/detection_ped_{trial_id}.json'
        if vehicle_id == None:
            return f'{data_dir}/dets/edge_DSRC_BSM_send_{trial_id}.csv'
        else:
            return f'{data_dir}/dets/edge_DSRC_BSM_send_{trial_id}_{vehicle_id}.csv'
    elif system == 'MSight':
        if object == "ped":
            return f'{data_dir}/dets/detection_ped_{trial_id}.json'
        if vehicle_id == None:
            return f'{data_dir}/dets/Test{trial_id}_MsightObjectList.csv'
        else:
            return f'{data_dir}/dets/Test{trial_id}_MsightObjectList_{vehicle_id}.csv'

# def sort_point_list(point_list):
#     return sorted(point_list, key=lambda x: x.time)


def interpolate(t1, t2, t2_next, lat2, lon2, lat2_next, lon2_next):
    # Calculate the fraction of the total time that has passed since t2
    fraction = (t1 - t2) / (t2_next - t2)

    # Use this fraction to calculate the interpolated latitude and longitude
    lat1 = lat2 + fraction * (lat2_next - lat2)
    lon1 = lon2 + fraction * (lon2_next - lon2)

    return lat1, lon1


def prepare_gt_data(vehicle1_data, vehicle2_data, cfg):
    veh1_sample_rate = cfg['GROUND_TRUTH_FREQUENCY_VEH1']
    veh2_sample_rate = cfg['GROUND_TRUTH_FREQUENCY_VEH2']
    if veh1_sample_rate < veh2_sample_rate:
        tmp = vehicle1_data
        vehicle1_data = vehicle2_data
        vehicle2_data = tmp
        tmp = veh1_sample_rate
        veh1_sample_rate = veh2_sample_rate
        veh2_sample_rate = tmp

    # print(vehicle1_data.lats, vehicle1_data.longs, vehicle1_data.datetime)
    p2 = 0
    dp_list = []
    for p1, t1 in enumerate(vehicle1_data.datetime):
        if p2 + 1 > len(vehicle2_data.datetime) - 1:
            break
        while int(vehicle2_data.datetime[p2+1]) < int(t1):
            p2 += 1
            if p2 + 1 > len(vehicle2_data.datetime) - 1:
                break
        if p2 + 1 > len(vehicle2_data.datetime) - 1:
            break
        t1 = int(t1)
        lat1 = vehicle1_data.lats[p1]
        lon1 = vehicle1_data.longs[p1]
        t2 = int(vehicle2_data.datetime[p2])
        lat2 = vehicle2_data.lats[p2]
        lon2 = vehicle2_data.longs[p2]
        if t1 < t2:
            continue
        t2_next = int(vehicle2_data.datetime[p2 + 1])
        lat2_next = vehicle2_data.lats[p2 + 1]
        lon2_next = vehicle2_data.longs[p2 + 1]
        lat2_interpolated, lon2_interpolated = interpolate(
            t1, t2, t2_next, lat2, lon2, lat2_next, lon2_next)
        point1 = DataPoint(id=0,
                           time=t1,
                           lat=lat1,
                           lon=lon1,
                           speed=0)
        point2 = DataPoint(id=1,
                           time=t1,
                           lat=lat2_interpolated,
                           lon=lon2_interpolated,
                           speed=0)
        dp_list.append(point1)
        dp_list.append(point2)

    ts = TrajectorySet(dp_list)
    for t in ts.trajectories.values():
        compute_speed_and_heading(t.dp_list, interval=10)
    return ts


class Result:
    def __init__(self) -> None:
        self.trials = []
        self._latency = None

    def add_trial(self, trial):
        assert 'id' in trial and 'type' in trial, 'trial should have id and type'
        self.trials.append(trial)

    @property
    def latency(self):
        if self._latency is not None:
            return self._latency
        return self.calc_mean('latency', kind='latency')

    def calc_mean(self, att, kind=None):
        s = 0
        n = 0
        for t in self.trials:
            if kind is not None and t['type'] != kind:
                continue
            if t[att] is not None:
                s += t[att]
                n += 1
        return s / n if n > 0 else None

    def has_type(self, kind):
        for t in self.trials:
            if t['type'] == kind:
                return True
        return False

    def to_tables(self):
        s = ''
        if self.has_type('latency'):
            s += f'+++++++++++++++++++Latency Results+++++++++++++++++++\n'
            headers = ['Trial ID', 'Latency']
            data = []
            for t in self.trials:
                if t['type'] == 'latency':
                    data.append([t['id'], t['latency']])
            data.append(['Mean', self.latency])
            s += tabulate(data, headers=headers, floatfmt=".3f")
            s += '\n'
        if self.has_type('vehicle_evaluation'):
            s += f'+++++++++++++++++++Vehicle Evaluation Results+++++++++++++++++++\n'
            headers = ['ID', 'FP Rate', 'FN Rate',
                       'MOTA', 'MOTP', 'IDF1', 'HOTA', 'IDS', 'freq', 'int var']
            data = []
            for t in self.trials:
                if t['type'] == 'vehicle_evaluation':
                    data.append([t['id'], t['fp_rate'],t['fnr_freq'],  t['mota_freq'], t['motp'],
                                t['idf1'], t['hota'], t['ids'], t['frequency'], t['interval_variance']])
            data.append(['Mean', self.calc_mean("fp_rate", "vehicle_evaluation"), self.calc_mean("fnr_freq", "vehicle_evaluation"), self.calc_mean("mota_freq", 'vehicle_evaluation'), self.calc_mean(
                "motp", "vehicle_evaluation"), self.calc_mean("idf1", "vehicle_evaluation"), self.calc_mean("hota", "vehicle_evaluation"), self.calc_mean("ids", "vehicle_evaluation")])
            s += tabulate(data, headers=headers, floatfmt=".3f")
            s += '\n'
        if self.has_type('pedestrian_evaluation'):
            s += f'+++++++++++++++++++Pedestrian Evaluation Results+++++++++++++++++++\n'
            headers = ['ID', 'FP Rate', 'FN Rate',
                       'MOTA', 'MOTP', 'IDF1', 'HOTA', 'IDS', 'freq', 'int var']
            data = []
            for t in self.trials:
                if t['type'] == 'pedestrian_evaluation':
                    data.append([t['id'], t['fp_rate'], t['fnr_freq'], t['mota_freq'], t['motp'],
                                t['idf1'], t['hota'], t['ids'], t['frequency'], t['interval_variance']])
            data.append(['Mean', self.calc_mean("fp_rate", "pedestrian_evaluation"), self.calc_mean("fnr_freq", "pedestrian_evaluation"), self.calc_mean("mota_freq", 'pedestrian_evaluation'), self.calc_mean(
                "motp", "pedestrian_evaluation"), self.calc_mean("idf1", "pedestrian_evaluation"), self.calc_mean("hota", "pedestrian_evaluation"), self.calc_mean("ids", "pedestrian_evaluation")])
            s += tabulate(data, headers=headers, floatfmt=".3f")
            s += '\n'
        return s

    def __str__(self) -> str:
        return self.to_tables()
