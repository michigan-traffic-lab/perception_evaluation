'''
Legacy data processing code to process the input GPS and detection data
'''
import datetime
import csv
import numpy as np
import pandas as pd
from os.path import exists
import json



class ExperimentData:
    def __init__(self):
        self.longs = []
        self.lats = []
        self.datetime = []
        self.texts = []
        self.speeds = []
        self.exist = True


class CAVData(ExperimentData):
    def __init__(self, data_files, plot_name, gps_type='oxford'):
        super().__init__()
        self.files = data_files
        self.gps_type = gps_type
        self.plot_name = plot_name
        if self.gps_type == 'oxford':
            assert type(
                data_files) is list, 'for oxford gps type, must pass both basic and speed data file as a list of length two'
            for file in data_files:
                if exists(file) == False:
                    # self.exist = False
                    pass
        else:
            if exists(data_files) == False:
                self.exist = False
        if self.exist:
            self.load_data_from_csv()

    def load_data_from_csv(self):
        if self.gps_type == 'oxford':
            with open(self.files[0]) as cav_traj:
                for row in csv.DictReader(cav_traj, delimiter=','):
                    self.add_basic_data(row)
            # with open(self.files[1]) as cav_speed:
            #     for row in csv.DictReader(cav_speed, delimiter=','):
            #         self.add_speed_data(row)
        elif self.gps_type == 'avalon':
            with open(self.files) as cav_traj:
                for row in csv.reader(cav_traj, delimiter=','):
                    self.add_basic_data(row)
        elif self.gps_type == 'oxford-new':
            with open(self.files) as cav_traj:
                for row in csv.DictReader(cav_traj, delimiter=','):
                    self.add_basic_data(row)

        else:
            raise Exception('GPS type error')

    def add_basic_data(self, row):
        if self.gps_type == 'oxford':
            self.datetime.append(row['field.header.stamp'])
            self.longs.append(float(row['field.longitude']))
            self.lats.append(float(row['field.latitude']))
            datetime_mill = datetime.datetime.fromtimestamp(
                float(row['field.header.stamp']) / 1000000000).strftime('%Y-%m-%d %H:%M:%S.%f')
            self.texts.append("UTC Time from RTK: " +
                              str(datetime_mill) + '<br>')
        elif self.gps_type == 'avalon':
            self.lats.append(float(row[11]))
            self.longs.append(float(row[12]))
            gpsweek = float(row[5])
            gpsseconds = float(row[6]) - 18 - 4 * 60 * 60
            UTCtime = pd.to_datetime(
                "1980-01-06 00:00:00") + pd.Timedelta(weeks=gpsweek) + pd.Timedelta(seconds=gpsseconds)
            str_UTCtime = UTCtime.strftime('%Y-%m-%d %H:%M:%S.%f')
            self.datetime.append(datetime.datetime.strptime(
                str_UTCtime, '%Y-%m-%d %H:%M:%S.%f').timestamp()*1000000000)
            self.texts.append("UTC Time from RTK: " + str(UTCtime))
        elif self.gps_type == 'oxford-new':
            self.datetime.append(float(row['secMarkDecoded'])*1000000000)
            self.longs.append(float(row['Long.']))
            self.lats.append(float(row['Lat.']))
            datetime_mill = datetime.datetime.fromtimestamp(
                float(row['secMarkDecoded'])).strftime('%Y-%m-%d %H:%M:%S.%f')
            self.texts.append("UTC Time from RTK: " +
                              str(datetime_mill) + '<br>')

    def add_speed_data(self, row):
        lat_speeds, long_speeds = row['field.twist.twist.linear.x'], row['field.twist.twist.linear.y']
        try:
            sequence = self.datetime.index(row['field.header.stamp'])
            self.speeds.append(
                ((float(lat_speeds)**2) + (float(long_speeds)**2)) ** 0.5)
            self.texts[sequence] = str(
                self.texts[sequence]) + 'Speed from RTK: ' + str(round(self.speeds[-1], 3)) + ' m/s'
        except:
            pass


class DetectionData(ExperimentData):
    def __init__(self, data_files, plot_name, detection_type):
        super().__init__()
        self.detection_type = detection_type
        self.plot_name = plot_name
        self.obj_ids = []
        self.msg_count = []
        self.obj_type = []

        self.files = data_files

        if exists(self.files) == False:
            self.exist = False

        if self.exist:
            self.load_data_from_csv()

    def load_data_from_csv(self):
        if self.detection_type == 'Derq':
            with open(self.files) as det_traj:
                for row in csv.DictReader(det_traj, delimiter=','):
                    self.add_detection_data(row)
        elif self.detection_type == 'Conti':
            with open(self.files) as det_traj:
                for row in csv.DictReader(det_traj, delimiter=';'):
                    self.add_detection_data_conti(row)
        elif self.detection_type == 'MSight':
            with open(self.files) as det_traj:
                for row in csv.DictReader(det_traj):
                    self.add_detection_data_Msight(row)
        elif self.detection_type == 'Bluecity':
            with open(self.files) as det_traj:
                obj_list = json.load(det_traj)
                for obj in obj_list:
                    self.add_detection_data_Bluecity(obj)

    def add_detection_data(self, row):
        self.datetime.append(float(datetime.datetime.strptime(
            row['UTC Timestamp'], '%Y-%m-%d %H:%M:%S.%f').timestamp() * 1000000000))
        self.longs.append(float(row['lon'])/10000000)
        self.lats.append(float(row['lat'])/10000000)
        self.speeds.append(float(row['speed'])/50)
        self.obj_ids.append(row['id'])
        self.msg_count.append(row['msgCnt'])
        self.texts.append("UTC Time from GPS: " + row['UTC Timestamp'] +
                          'Speed from CI: ' + str(round(self.speeds[-1], 3)) + ' m/s')

    def add_detection_data_conti(self, row):
        RSU_ref = {'lat': 42.300790, 'lon': -83.698575}
        obj_count = int(
            row['pb.SmartCities.sensorDataSharingMessage.objects.count'])
        r_earth = 6378000  # earth radius 6378 km
        for obj in range(10):
            if f'pb.SmartCities.sensorDataSharingMessage.objects.objectData.{obj}.detObjCommon.pos.offsetY' in row.keys():
                out_of_range = False
                try:
                    lat = RSU_ref['lat'] + (float(
                        row[f'pb.SmartCities.sensorDataSharingMessage.objects.objectData.{obj}.detObjCommon.pos.offsetY']) / r_earth) * (180 / np.pi)
                    long = RSU_ref['lon'] + (float(row[f'pb.SmartCities.sensorDataSharingMessage.objects.objectData.{obj}.detObjCommon.pos.offsetX']) / r_earth) * (
                        180 / np.pi) / np.cos(RSU_ref['lat'] * np.pi/180)
                except:
                    # if either lat or long is missing, provide a lat lon in Mcity but away from the ROI
                    lat, long = 42.301004, -83.697832
                    out_of_range = True
                # if not out_of_range:
                    # print(row[f'pb.SmartCities.sensorDataSharingMessage.objects.objectData.{obj}.detObjCommon.speedConfidence'])
                # out_of_range=False
                datetime_mill = float(row['sender_timestamp']) / 1000000
                raise NotImplementedError
                self.datetime.append(datetime_mill)
                self.longs.append(long)
                self.lats.append(lat)
                try:
                    if row[f'pb.SmartCities.sensorDataSharingMessage.objects.objectData.{obj}.detObjCommon.speed'] == '':
                        self.speeds.append(float(0))  # float(row['speed'])/50
                    else:
                        self.speeds.append(float(
                            row[f'pb.SmartCities.sensorDataSharingMessage.objects.objectData.{obj}.detObjCommon.speed']))
                except:
                    self.speeds.append(float(0))
                # the comment scripts are used to check whether the pcap file and csv file are consistent or not
                # if row[f'pb.SmartCities.sensorDataSharingMessage.objects.objectData.{obj}.detObjCommon.objectID'] == '2298':
                #     if row[f'pb.SmartCities.sensorDataSharingMessage.objects.objectData.{obj}.detObjCommon.heading'] == '269':
                #         iii=1

                self.obj_ids.append(
                    row[f'pb.SmartCities.sensorDataSharingMessage.objects.objectData.{obj}.detObjCommon.objectID'])
                self.obj_type.append(
                    row[f'pb.SmartCities.sensorDataSharingMessage.objects.objectData.{obj}.detObjCommon.objType'])
                # self.msg_count.append(row['msgCnt'])
                self.texts.append("UTC Time from GPS: " + datetime_mill +
                                  'Speed from CI: ' + str(round(self.speeds[-1], 3)) + ' m/s')

    def add_detection_data_Msight(self, row):
        lat = float(row['Lat.'])
        long = float(row['Long.'])
        self.datetime.append(float(datetime.datetime.strptime(
            row['secMarkDecoded'], '%Y-%m-%d %H-%M-%S-%f').timestamp() * 1000000000))
        self.longs.append(long)
        self.lats.append(lat)
        if row['Speed'] == '':
            self.speeds.append(0)
        else:
            self.speeds.append(float(row['Speed']))
        self.obj_ids.append(row['Vehicle ID'])
        self.msg_count.append(row['msgCountDecoded'])
        self.texts.append("UTC Time from GPS: " + row['secMarkDecoded'] +
                          'Speed from CI: ' + str(round(self.speeds[-1], 3)) + ' m/s')

    def add_detection_data_Bluecity(self, row):
        if len(row['objs']) != 0:
            for obj in row['objs']:
                current_t = row['timestamp'] * 1e9
                # print(current_t, obj['id'])
                lat = obj['lat']
                long = obj['lon']
                self.datetime.append(current_t)
                self.longs.append(long)
                self.lats.append(lat)
                self.speeds.append(float(obj['speed']))
                self.obj_ids.append(obj['id'])
                self.obj_type.append(obj['classType'])
