import numpy as np
from datetime import datetime
from geopy import distance as geodistance


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
        self.match = -1
        self.speed = speed
        self.heading = None
        self.lat_error = None
        self.lon_error = None
        self.tp = False
        self.timestr = datetime.fromtimestamp(time / 1000000000)

    def __str__(self):
        return f'UTC time: {self.timestr}, ID: {self.id}, speed: {self.speed}, heading: {self.heading}'

class DataFrame:
    pass

class TrajectorySet:
    def query_by_frame():
        pass
    
    def query_by_id():
        pass


def distance(lat1, lon1, lat2, lon2):
    # dx, dy = (lat2 - lat1) * 111000., (lon2 - lon1) * 111000. * np.cos(lat2/180*np.pi)
    dx = geodistance.geodesic([lat2, lon1], [lat1, lon1]).m * np.sign(lat2 - lat1)
    dy = geodistance.geodesic([lat2, lon2], [lat2, lon1]).m * np.sign(lon2 - lon1)
    return dx, dy

def compute_speed_and_heading(dp_list, interval=2):
    '''
    interval in each direction (past, future)
    '''
    for i in range(len(dp_list)):
        dp = dp_list[i]
        past_t = max(0, i-interval)
        future_t = min(len(dp_list)-1, i+interval)
        dx, dy = distance(dp_list[past_t].lat, dp_list[past_t].lon, dp_list[future_t].lat, dp_list[future_t].lon)

        # compute speed m/s
        dp.speed = (dx**2 + dy**2)**0.5 / (dp_list[future_t].time - dp_list[past_t].time) * 1000000000
        
        # compute heading
        # clock-wise, in degree
        # north: 0, east: pi/2, north-east: pi/4
        dp.heading = np.arctan2(dy, dx)


def prepare_data(dt, gt):

    # convert detections to datapoints
    dtdp_list = []
    for i in range(len(dt.lats)):
        # dt_time = float(datetime.strptime(dt.datetime[i], '%Y-%m-%d %H-%M-%S-%f').timestamp() * 1000000000)
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

    return dtdp_list, gtdp_list