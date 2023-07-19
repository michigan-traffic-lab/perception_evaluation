import numpy as np
from geopy import distance as geodistance
from datetime import datetime
import pandas as pd


class Trajectory:
    def __init__(self, dp_list, id=None, sample_rate=None) -> None:
        self.id = id
        self.dp_list = dp_list
        self.sample_rate = sample_rate
        self.match = None

    def add_dp(self, dp):
        if dp.id != self.id:
            raise ValueError('id not match')
        self.dp_list.append(dp)


class DataPoint:
    def __init__(self, id, time, lat, lon, speed=None, conf=1) -> None:
        '''
        id: the object id
        time: timestamp of the data point
        conf: detection confidence score
        match: the match index (-1 means not matched to anyone)
        speed: the object speed (m/s)
        heading: object speed heading (-pi, pi), clockwise, north as 0
        lat_error: lateral error w.r.t object heading, signed
        lon_error: longitudinal error w.r.t object heading, signed
        tp: whether it is a true positive detection
        timestr: time string converted from timestamp
        '''
        self.id = id
        self.time = time
        self.lat = lat
        self.lon = lon
        self.conf = conf
        self.point_wise_match = None
        self.traj_wise_match = None
        self.speed = speed
        self.heading = None
        self.lat_error = None
        self.lon_error = None
        self.tp = False
        # print(time, 'time in datapoint')
        self.timestr = datetime.fromtimestamp(time / 1e9)

    def __str__(self):
        return f'DataPoint(UTC time: {self.timestr}, ID: {self.id}, speed: {self.speed}, heading: {self.heading}, match: {self.match}, lat_error: {self.lat_error}, lon_error: {self.lon_error})'

    def __repr__(self) -> str:
        return self.__str__()

    @property
    def error(self):
        if self.lat_error is None or self.lon_error is None:
            return None
        return (self.lat_error ** 2 + self.lon_error ** 2) ** 0.5


class DataFrame:
    def __init__(self, dp_list, time=None) -> None:
        self.dp_list = dp_list
        self.time = time

    def add_dp(self, dp):
        if dp.time != self.time:
            raise ValueError('time not match')
        self.dp_list.append(dp)


class TrajectorySet:
    def __init__(self, dp_list) -> None:
        self.dp_list = dp_list
        self.trajectories = {}  # id: trajectory
        self.dataframes = []  # list of dataframe
        # print('++++++++++++++++++++++++++++++++++++++')
        previous_time = None
        for dp in self.dp_list:
            # print(dp.time, 'time in trajectoryset ', dp.id)
            t = int(dp.time)  # note that we will use int to represent time
            if not t == previous_time:
                self.dataframes.append(DataFrame(dp_list=[dp], time=t))
                previous_time = t
            if dp.id not in self.trajectories:
                self.trajectories[dp.id] = Trajectory(dp_list=[dp], id=dp.id)
            else:
                self.trajectories[dp.id].add_dp(dp)

    @property
    def ids(self):
        return list(self.trajectories.keys())

    @property
    def num_traj(self):
        return len(self.trajectories)

    def query_by_frame():
        pass

    def query_by_id():
        pass


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


def nested_dict_to_matrix(nested_dict):
    # Convert the nested dictionary to a DataFrame
    df = pd.DataFrame(nested_dict)

    # Convert the DataFrame to a matrix
    matrix = df.values

    return matrix


# def matrix_to_nested_dict(matrix, nested_dict_template):
#     # Get the row keys (second level keys) and column keys (first level keys)
#     row_keys = list(nested_dict_template.values())[0].keys()
#     col_keys = nested_dict_template.keys()

#     # Convert the matrix back to a nested dictionary
#     nested_dict_again = {col: {row: matrix[i][j] for i, row in enumerate(
#         row_keys)} for j, col in enumerate(col_keys)}

#     return nested_dict_again
