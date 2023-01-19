import plotly.graph_objects as go
import pathlib
import datetime
import csv
import numpy as np
import argparse
from pathlib import Path
import pandas as pd
import yaml
from os.path import exists
from geopy import distance
import geopy
import json
import scipy.interpolate as interpolate

parser = argparse.ArgumentParser()
parser.add_argument('--trial-name', default='5-2')
parser.add_argument('--experiment-name', default='01-05-2023')
parser.add_argument('--config-file', '-c', help="configuration file path", default='experiment_settings.yml')
args = parser.parse_args()

with open(args.config_file, 'r') as f:
    yaml_settings = yaml.safe_load(f)

RADIUS = yaml_settings['radius']
CENTER_COR=yaml_settings['AOI-center-coordination']

if yaml_settings['data-dir'] is None:
    HomeDir = Path.home()
    if yaml_settings['system'] == 'linux':
        cav_1_rtk_file = HomeDir / 'Archive/TestingResult' / args.experiment_name / str(args.trial_name) / f'gps_fix_{args.trial_name}.txt'
        cav_1_speed_file = HomeDir / 'Archive/TestingResult' / args.experiment_name / str(args.trial_name) / f'gps_vel_{args.trial_name}.txt'
        cav_2_rtk_file = HomeDir / 'Archive/TestingResult' / args.experiment_name / str(args.trial_name) / f'gps_veh2_{args.trial_name}.gps'
        det_traj_file = HomeDir / 'Archive/TestingResult' / args.experiment_name / str(args.trial_name) / f'edge_DSRC_BSM_send_{args.name}.csv'
    else:
        cav_1_rtk_file = str(HomeDir) + str(pathlib.PureWindowsPath('/Documents/GitHub/Archive/TestingResult/' + args.experiment_name + '/' + str(args.trial_name) + '/gps_fix_' + str(
            args.trial_name) + '.txt'))
        cav_1_speed_file = str(HomeDir) + str(pathlib.PureWindowsPath('/Documents/GitHub//Archive/TestingResult/' + args.experiment_name + '/' + str(args.trial_name) + '/gps_vel_' + str(
            args.trial_name) + '.txt'))

        if yaml_settings['vehicles'][1]['gps_type']=='oxford':
            cav_2_rtk_file = str(HomeDir) + str(pathlib.PureWindowsPath('/Documents/GitHub//Archive/TestingResult/' + args.experiment_name + '/' + str(args.trial_name) + '/gps_fix_veh2_' + str(
                args.trial_name) + '.txt'))
            cav_2_speed_file = str(HomeDir) + str(pathlib.PureWindowsPath('/Documents/GitHub//Archive/TestingResult/' + args.experiment_name + '/' + str(args.trial_name) + '/gps_vel_veh2_' + str(
            args.trial_name) + '.txt'))
        elif yaml_settings['vehicles'][1]['gps_type']=='oxford-new':
            cav_2_rtk_file = str(HomeDir) + str(pathlib.PureWindowsPath('/Documents/GitHub//Archive/TestingResult/' + args.experiment_name + '/' + str(args.trial_name) + '/gps_veh2_' + str(
                args.trial_name) + '.csv'))
        elif yaml_settings['vehicles'][1]['gps_type']=='avalon':
            cav_2_rtk_file = str(HomeDir) + str(pathlib.PureWindowsPath('/Documents/GitHub//Archive/TestingResult/' + args.experiment_name + '/' + str(args.trial_name) + '/gps_veh2_' + str(
                args.trial_name) + '.gps'))
        det_traj_file = str(HomeDir) + str(pathlib.PureWindowsPath('/Documents/GitHub//Archive/TestingResult/' + args.experiment_name + '/' + str(args.trial_name) + '/edge_DSRC_BSM_send_' + str(
            args.trial_name) + '.csv'))
        det_traj_file2 = str(HomeDir) + str(pathlib.PureWindowsPath('/Documents/GitHub//Archive/TestingResult/' + args.experiment_name + '/' + 'Mcity_Tests/Exports/Test' + str(
            args.trial_name) + '_SDSMTx.csv'))
        det_traj_file3 = str(HomeDir) + str(pathlib.PureWindowsPath('/Documents/GitHub//Archive/TestingResult/' + args.experiment_name + '/' + 'Msight/Test' + str(
            args.trial_name) + '_MsightObjectList.csv'))
        det_traj_file4 = str(HomeDir) + str(pathlib.PureWindowsPath('/Documents/GitHub//Archive/TestingResult/' + args.experiment_name + '/' + 'Bluecity/detection_' + str(
            args.trial_name) + '.json'))
        det_ped_file = str(HomeDir) + str(pathlib.PureWindowsPath('/Documents/GitHub//Archive/TestingResult/' + args.experiment_name + '/' + str(args.trial_name) + '/ped_' + str(
            args.trial_name) + '.txt'))
else:
    cav_1_rtk_file = Path(yaml_settings['data-dir'])  / args.name / f'gps_fix_{args.name}.txt'
    det_traj_file = Path(yaml_settings['data-dir'])  / args.name / f'DSRC_BSM_server_receive_{args.name}.csv'
    cav_2_rtk_file = Path(yaml_settings['data-dir'])  / args.name / f'gps_veh2_{args.name}.gps'
    cav_1_speed_file = Path(yaml_settings['data-dir'])  / args.name / f'gps_vel_{args.name}.gps'

class ExperimentData:
    def __init__(self):
        self.longs = []
        self.lats = []
        self.datetime = []
        self.texts = []
        self.speeds = []
        self.exist = True

class PedData(ExperimentData):
    def __init__(self, data_files,plot_name, gps_type='ublox'):
        super().__init__()
        self.files = data_files
        self.gps_type=gps_type
        self.plot_name=plot_name
        if self.gps_type == 'ublox':
            if exists(data_files) == False:
                self.exist = False
        if self.exist:
            self.load_data_from_csv()
        
        #self.interpolate_points()

    def load_data_from_csv(self):
        if self.gps_type == 'ublox':
            with open(self.files) as det_traj:
                for row in csv.reader(det_traj):
                    self.add_ped_data(row)

    def add_ped_data(self,row):
        lat=float(row[1])
        long=float(row[2])
        out_of_range=filter_distance_to_center({'lat':lat,'lon':long})

        if not out_of_range:
            hour=str(int(row[0][:2])-5)
            minute=row[0][2:4]
            second=f'{row[0][4:]}0000'
            date=f'{args.experiment_name[-4:]}-{args.experiment_name[0:5]}'
            self.datetime.append(f'{date} {hour}:{minute}:{second}')
            self.longs.append(long)
            self.lats.append(lat)
            self.texts.append("UTC Time from GPS: " + self.datetime[-1])
    
    def interpolate_points(self):
        rst=[]
        num_interval_interpolate=2
        length_interval=0.1 # 0.1 seconds
        for t in range(len(self.lats)-1):
            x0, y0= self.lats[t], self.longs[t]
            x1, y1= self.lats[t+1], self.longs[t+1]
            datetime_mill = datetime.datetime.strptime(self.datetime[t],'%Y-%m-%d %H:%M:%S.%f').timestamp()*1000000000
            for i in range(num_interval_interpolate):
                x_increase=(x1-x0)/num_interval_interpolate*i
                y_increase=(y1-y0)/num_interval_interpolate*i
                time_increase=i*length_interval/num_interval_interpolate*1000000000
                new_date= datetime.datetime.fromtimestamp((datetime_mill+time_increase)/ 1000000000).strftime('%Y-%m-%d %H:%M:%S.%f')
                texts=f'UTC Time from GPS: {new_date}'
                rst.append([new_date,x0+x_increase,y0+y_increase,texts])
        self.datetime=[i[0] for i in rst]
        self.lats=[i[1] for i in rst]
        self.longs=[i[2] for i in rst]
        self.texts=[i[3] for i in rst]

class CAVData(ExperimentData):
    def __init__(self, data_files,plot_name, gps_type='oxford'):
        super().__init__()
        self.files = data_files
        self.gps_type=gps_type
        self.plot_name=plot_name
        if self.gps_type == 'oxford':
            assert type(data_files) is list, 'for oxford gps type, must pass both basic and speed data file as a list of length two'
            for file in data_files:
                if exists(file) == False:
                    self.exist = False
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
            with open(self.files[1]) as cav_speed:
                for row in csv.DictReader(cav_speed, delimiter=','):
                    self.add_speed_data(row)
        elif self.gps_type == 'avalon' :
            with open(self.files) as cav_traj:
                for row in csv.reader(cav_traj, delimiter=','):
                    self.add_basic_data(row)
        elif self.gps_type == 'oxford-new' :
            with open(self.files) as cav_traj:
                for row in csv.DictReader(cav_traj, delimiter=','):
                    self.add_basic_data(row)

        else:
            raise Exception('GPS type error')
        
    def add_basic_data(self, row):
        if self.gps_type == 'oxford':
            lat=float(row['field.latitude'])
            long=float(row['field.longitude'])
            out_of_range=filter_distance_to_center({'lat':lat,'lon':long})

            if not out_of_range:
                self.datetime.append(row['field.header.stamp'])
                self.longs.append(float(row['field.longitude']))
                self.lats.append(float(row['field.latitude']))
                datetime_mill = datetime.datetime.fromtimestamp(float(row['field.header.stamp']) / 1000000000).strftime('%Y-%m-%d %H:%M:%S.%f')
                self.texts.append("UTC Time from RTK: " + str(datetime_mill) + '<br>')
        elif self.gps_type == 'avalon':
            lat=float(row[11])
            long=float(row[12])
            out_of_range=filter_distance_to_center({'lat':lat,'lon':long})
            
            if not out_of_range:
                self.lats.append(float(row[11]))
                self.longs.append(float(row[12]))
                gpsweek = float(row[5])
                gpsseconds = float(row[6]) - 18 - 4 * 60 * 60
                UTCtime = pd.to_datetime("1980-01-06 00:00:00") + pd.Timedelta(weeks=gpsweek) + pd.Timedelta(seconds=gpsseconds)
                str_UTCtime=UTCtime.strftime('%Y-%m-%d %H:%M:%S.%f')
                self.datetime.append(datetime.datetime.strptime(str_UTCtime,'%Y-%m-%d %H:%M:%S.%f').timestamp()*1000000000)
                self.texts.append("UTC Time from RTK: " + str(UTCtime))
        elif self.gps_type == 'oxford-new':
            lat=float(row['Lat.'])
            long=float(row['Long.'])
            out_of_range=filter_distance_to_center({'lat':lat,'lon':long})
            
            if not out_of_range:
                self.datetime.append(float(row['secMarkDecoded'])*1000000000)
                self.longs.append(float(row['Long.']))
                self.lats.append(float(row['Lat.']))
                datetime_mill = datetime.datetime.fromtimestamp(float(row['secMarkDecoded'])).strftime('%Y-%m-%d %H:%M:%S.%f')
                self.texts.append("UTC Time from RTK: " + str(datetime_mill) + '<br>')


    def add_speed_data(self, row):
        lat_speeds, long_speeds = row['field.twist.twist.linear.x'], row['field.twist.twist.linear.y']
        try:
            sequence=self.datetime.index(row['field.header.stamp'])
            self.speeds.append(((float(lat_speeds)**2) + (float(long_speeds)**2)) ** 0.5)
            self.texts[sequence] = str(self.texts[sequence]) + 'Speed from RTK: ' + str(round(self.speeds[-1], 3)) + ' m/s'
        except:
            pass
    
        
class DetectionData(ExperimentData):
    def __init__(self,data_files,plot_name, detection_type,plot_flag):
        super().__init__()
        self.detection_type=detection_type
        self.plot_name=plot_name
        self.plot_flag=plot_flag
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
                    self.add_detection_data_derq(row)
        elif self.detection_type == 'Conti':
            with open(self.files) as det_traj:
                for row in csv.DictReader(det_traj, delimiter=';'):
                    self.add_detection_data_conti(row)
        elif self.detection_type == 'Msight':
            with open(self.files) as det_traj:
                for row in csv.DictReader(det_traj):
                    self.add_detection_data_Msight(row)
        elif self.detection_type == 'Bluecity':
            with open(self.files) as det_traj:
                obj_list=json.load(det_traj)
                for obj in obj_list:
                    self.add_detection_data_Bluecity(obj)

    def add_detection_data_derq(self, row):
        lat=float(row['lat'])/10000000
        long=float(row['lon'])/10000000
        out_of_range=filter_distance_to_center({'lat':lat,'lon':long})

        if not out_of_range:
            self.datetime.append(row['UTC Timestamp'])
            self.longs.append(float(row['lon'])/10000000)
            self.lats.append(float(row['lat'])/10000000)
            self.speeds.append(float(row['speed'])/50)
            self.obj_ids.append(row['id'])
            self.msg_count.append(row['msgCnt'])
            self.texts.append("UTC Time from GPS: " + row['UTC Timestamp'] + 'Speed from CI: ' + str(round(self.speeds[-1], 3)) + ' m/s')

    def add_detection_data_conti(self, row):
        #RSU_ref={'lat':42.30086635,'lon':-83.698339820000001}
        RSU_ref={'lat':42.300790,'lon':-83.698575}
        obj_count=int(row['pb.SmartCities.sensorDataSharingMessage.objects.count'])
        r_earth = 6378000 # earth radius 6378 km
        for obj in range(10):
            if f'pb.SmartCities.sensorDataSharingMessage.objects.objectData.{obj}.detObjCommon.pos.offsetY' in row.keys():
                out_of_range=False
                try:
                    start = geopy.Point(longitude=RSU_ref['lon'],latitude=RSU_ref['lat'])
                    move1=distance.distance(meters=float(row[f'pb.SmartCities.sensorDataSharingMessage.objects.objectData.{obj}.detObjCommon.pos.offsetX']))
                    move2=distance.distance(meters=float(row[f'pb.SmartCities.sensorDataSharingMessage.objects.objectData.{obj}.detObjCommon.pos.offsetY']))
                    point1=move1.destination(point=start, bearing=90)
                    point2=move2.destination(point=point1, bearing=0)
                    lat, long=point2[0], point2[1]
                    # lat=RSU_ref['lat'] + (float(row[f'pb.SmartCities.sensorDataSharingMessage.objects.objectData.{obj}.detObjCommon.pos.offsetY']) / r_earth) * (180 / np.pi)
                    # long=RSU_ref['lon'] + (float(row[f'pb.SmartCities.sensorDataSharingMessage.objects.objectData.{obj}.detObjCommon.pos.offsetX']) / r_earth) * (180 / np.pi) / np.cos(RSU_ref['lat'] * np.pi/180)
                except:
                    lat, long=42.301004, -83.697832#if either lat or long is missing, provide a lat lon in Mcity but away from the ROI
                    out_of_range=True
                #if not out_of_range:
                    #print(row[f'pb.SmartCities.sensorDataSharingMessage.objects.objectData.{obj}.detObjCommon.speedConfidence'])
                out_of_range=filter_distance_to_center({'lat':lat,'lon':long})
                #out_of_range=False
                if not out_of_range:
                    datetime_mill = datetime.datetime.fromtimestamp(float(row['sender_timestamp']) / 1000000).strftime('%Y-%m-%d %H:%M:%S.%f')
                    self.datetime.append(datetime_mill)
                    self.longs.append(long)
                    self.lats.append(lat)
                    try:
                        if row[f'pb.SmartCities.sensorDataSharingMessage.objects.objectData.{obj}.detObjCommon.speed'] == '':
                            self.speeds.append(float(0))#float(row['speed'])/50
                        else:
                            self.speeds.append(float(row[f'pb.SmartCities.sensorDataSharingMessage.objects.objectData.{obj}.detObjCommon.speed']))
                    except:
                        self.speeds.append(float(0))
                    # the comment scripts are used to check whether the pcap file and csv file are consistent or not
                    # if row[f'pb.SmartCities.sensorDataSharingMessage.objects.objectData.{obj}.detObjCommon.objectID'] == '2298':
                    #     if row[f'pb.SmartCities.sensorDataSharingMessage.objects.objectData.{obj}.detObjCommon.heading'] == '269':
                    #         iii=1

                    self.obj_ids.append(row[f'pb.SmartCities.sensorDataSharingMessage.objects.objectData.{obj}.detObjCommon.objectID'])
                    self.obj_type.append(row[f'pb.SmartCities.sensorDataSharingMessage.objects.objectData.{obj}.detObjCommon.objType'])
                    #self.msg_count.append(row['msgCnt'])
                    self.texts.append("UTC Time from GPS: " + datetime_mill + 'Speed from CI: ' + str(round(self.speeds[-1], 3)) + ' m/s')
    
    def add_detection_data_Msight(self, row):
        lat=float(row['Lat.'])
        long=float(row['Long.'])
        out_of_range=filter_distance_to_center({'lat':lat,'lon':long})

        if not out_of_range:
            self.datetime.append(row['secMarkDecoded'])
            self.longs.append(long)
            self.lats.append(lat)
            if row['Speed']=='':
                self.speeds.append(0)
            else:
                self.speeds.append(float(row['Speed']))
            self.obj_ids.append(row['Vehicle ID'])
            self.msg_count.append(row['msgCountDecoded'])
            self.texts.append("UTC Time from GPS: " + row['secMarkDecoded'] + 'Speed from CI: ' + str(round(self.speeds[-1], 3)) + ' m/s')
    
    def add_detection_data_Bluecity(self, row):
        if len(row['objs']) != 0:
            for obj in row['objs']:
                current_t = datetime.datetime.fromtimestamp(row['timestamp']).strftime('%Y-%m-%d %H:%M:%S.%f')
                lat=obj['lat']
                long=obj['lon']
                out_of_range=filter_distance_to_center({'lat':lat,'lon':long})

                if not out_of_range:
                    self.datetime.append(current_t)
                    self.longs.append(long)
                    self.lats.append(lat)
                    self.speeds.append(float(obj['speed']))
                    self.obj_ids.append(obj['id'])
                    self.obj_type.append(obj['classType'])
                    self.texts.append("UTC Time from GPS: " + current_t + 'Speed from CI: ' + str(round(self.speeds[-1], 3)) + ' m/s')

class Plotter:
    def __init__(self):
        self.fig = None
        
    def plot_vehicle_data(self, data):
        if self.fig is None:
            self.fig=go.Figure(go.Scattermapbox(
                lat=data.lats, lon=data.longs, mode='markers', text=data.texts,
                marker={'size': 5, 'color': 'blue'}, name=data.plot_name))
        else:
            self.fig.add_trace(go.Scattermapbox(lat=data.lats, lon=data.longs, mode='markers', text=data.texts,
                marker={'size': 4, 'color': 'red'}, name=data.plot_name))
    
    def plot_ped_data(self, ped_data):
        self.fig.add_trace(go.Scattermapbox(
                    lon=ped_data.longs, lat=ped_data.lats, mode='markers', text=ped_data.texts,
                    marker={'size': 6, 'color': 'green'}, name='Ped trajectory recorded by GPS'))

    def plot_matching(self, gt_data,det_data,color):
        for i in range(len(gt_data.datetime)):
            if det_data[i]!=0:
                self.fig.add_trace(go.Scattermapbox(lat=[gt_data.lats[i],det_data[i][1]], lon=[gt_data.longs[i],det_data[i][2]], mode='lines',
                marker={'size': 10, 'color': color}, name='matching'))

    def plot_testing(self, pd_data):
        for i in pd_data[0]:
            self.fig.add_trace(go.Scattermapbox(lat=[i[0]], lon=[i[1]], mode='markers',
            marker={'size': 10, 'color': 'red'}, name='ped'))

    def plot_detection(self, data):
        for id in set(data.obj_ids):
            det_long_with_id = []
            det_lat_with_id = []
            det_speed_with_id = []
            det_text_with_id = []
            det_obj_type= []
            for i in range(len(data.obj_ids)):
                if data.obj_ids[i] == id:
                    det_long_with_id.append(data.longs[i])
                    det_lat_with_id.append(data.lats[i])
                    det_speed_with_id.append(data.speeds[i])
                    if len(data.obj_type) !=0:
                        det_obj_type.append(data.obj_type[i])
                    if data.detection_type=='Derq':
                        det_text_with_id.append("UTC time when detection result is received by edge: " + str(data.datetime[i]) + '' +
                                            '<br>' + 'Speed from Derq: ' + str(round(det_speed_with_id[-1], 3)) + ' m/s')
                    if data.detection_type=='Conti':
                        if data.obj_type[i]=='1':
                            det_text_with_id.append("UTC time when detection result is received by edge: " + str(data.datetime[i]) + '' +
                                                '<br>' + 'CAV Speed from Conti: ' + str(round(det_speed_with_id[-1], 3)) + ' m/s')
                        elif data.obj_type[i]=='2':
                            det_text_with_id.append("UTC time when detection result is received by edge: " + str(data.datetime[i]) + '' +
                                                '<br>' + 'Ped Speed from Conti: ' + str(round(det_speed_with_id[-1], 3)) + ' m/s')
                        elif data.obj_type[i]=='0':
                            det_text_with_id.append("UTC time when detection result is received by edge: " + str(data.datetime[i]) + '' +
                                                '<br>' + 'Unknown Speed from Conti: ' + str(round(det_speed_with_id[-1], 3)) + ' m/s')
                        else:
                            det_text_with_id.append("UTC time when detection result is received by edge: " + str(data.datetime[i]) + '' +
                                                '<br>' + 'Undefined Speed from Conti: ' + str(round(det_speed_with_id[-1], 3)) + ' m/s')
                    if data.detection_type=='Msight':
                        det_text_with_id.append("UTC time when detection result is received by edge: " + str(data.datetime[i]) + '' +
                                            '<br>' + 'Speed from Msight: ' + str(round(det_speed_with_id[-1], 3)) + ' m/s')
                    
                    if data.detection_type=='Bluecity':
                        if data.obj_type[i]=='2':
                            det_text_with_id.append("UTC time when detection result is received by edge: " + str(data.datetime[i]) + '' +
                                                '<br>' + 'CAV Speed from Bluecity: ' + str(round(det_speed_with_id[-1], 3)) + ' m/s')
                        elif data.obj_type[i]=='10':
                            det_text_with_id.append("UTC time when detection result is received by edge: " + str(data.datetime[i]) + '' +
                                                '<br>' + 'Ped Speed from Bluecity: ' + str(round(det_speed_with_id[-1], 3)) + ' m/s')
                        else:
                            det_text_with_id.append("UTC time when detection result is received by edge: " + str(data.datetime[i]) + '' +
                                                '<br>' + 'Undefined Speed from Bluecity: ' + str(round(det_speed_with_id[-1], 3)) + ' m/s')

            # if len(data.obj_type) !=0 and len(set(det_obj_type)) != 1:
            #     print('object type change error')

            
            if data.detection_type=='Derq':
                self.fig.add_trace(go.Scattermapbox(
                    lon=det_long_with_id, lat=det_lat_with_id, mode='markers', text=det_text_with_id,
                    marker={'size': 6, 'color': int(id)}, name='CAV trajectory ID:' + id + ' detected by Derq'))
            
            elif data.detection_type=='Msight':
                self.fig.add_trace(go.Scattermapbox(
                    lon=det_long_with_id, lat=det_lat_with_id, mode='markers', text=det_text_with_id,
                    marker={'size': 6, 'color': int(id)}, name='CAV trajectory ID:' + id + ' detected by Msight'))
            
            elif data.detection_type=='Conti':
                if data.obj_type[0]=='1':
                    self.fig.add_trace(go.Scattermapbox(
                        lon=det_long_with_id, lat=det_lat_with_id, mode='markers', text=det_text_with_id,
                        marker={'size': 6, 'color': int(id)}, name='CAV trajectory ID:' + id + ' detected by Conti'))
                elif data.obj_type[0]=='2':
                    self.fig.add_trace(go.Scattermapbox(
                        lon=det_long_with_id, lat=det_lat_with_id, mode='markers', text=det_text_with_id,
                        marker={'size': 6, 'color': int(id)}, name='Ped trajectory ID:' + id + ' detected by Conti'))
                elif data.obj_type[0]=='0':
                    self.fig.add_trace(go.Scattermapbox(
                        lon=det_long_with_id, lat=det_lat_with_id, mode='markers', text=det_text_with_id,
                        marker={'size': 6, 'color': int(id)}, name='Unknown trajectory ID:' + id + ' detected by Conti'))
                else:
                    self.fig.add_trace(go.Scattermapbox(
                            lon=det_long_with_id, lat=det_lat_with_id, mode='markers', text=det_text_with_id,
                            marker={'size': 6, 'color': int(id)}, name='Undefined trajectory ID:' + id + ' detected by Conti'))
            
            elif data.detection_type=='Bluecity':
                if det_obj_type[0]=='2':
                    self.fig.add_trace(go.Scattermapbox(
                        lon=det_long_with_id, lat=det_lat_with_id, mode='markers', text=det_text_with_id,
                        marker={'size': 6, 'color': int(id)}, name='CAV trajectory ID:' + id + ' detected by Bluecity'))
                elif det_obj_type[0]=='10':
                    self.fig.add_trace(go.Scattermapbox(
                        lon=det_long_with_id, lat=det_lat_with_id, mode='markers', text=det_text_with_id,
                        marker={'size': 6, 'color': int(id)}, name='Ped trajectory ID:' + id + ' detected by Bluecity'))
                else:
                    self.fig.add_trace(go.Scattermapbox(
                            lon=det_long_with_id, lat=det_lat_with_id, mode='markers', text=det_text_with_id,
                            marker={'size': 6, 'color': int(id)}, name='Undefined trajectory ID:' + id + ' detected by Bluecity'))
                
        self.fig.update_layout(
            mapbox={
                'accesstoken': 'pk.eyJ1Ijoiemh1aGoiLCJhIjoiY2ttdGF2bDd1MHEwbjJybzVxMTduOHVjbSJ9.wT_hzXAry0m33OrdaeTzRA',
                'style': "satellite-streets", 'center': {'lat': (np.mean(data.lats)), 'lon': (np.mean(data.longs))},
                'zoom': 20},
            showlegend=True, title_text='CAV Trajectories Recorded by RTK and Detected by CI')

# Area of interest filter
def filter_distance_to_center(data):
    d=distance.geodesic([data['lat'],data['lon']], CENTER_COR).m

    if d>RADIUS:
        out_of_range=True
    else:
        out_of_range=False
    return out_of_range


if __name__  == '__main__':
    vehicle_data = [
        CAVData([cav_1_rtk_file, cav_1_speed_file], plot_name=yaml_settings['vehicles'][0]['name'],gps_type=yaml_settings['vehicles'][0]['gps_type']),
        CAVData([cav_2_rtk_file, cav_2_speed_file], plot_name=yaml_settings['vehicles'][1]['name'],gps_type=yaml_settings['vehicles'][1]['gps_type']),
        #CAVData(cav_2_rtk_file, plot_name=yaml_settings['vehicles'][1]['name'],gps_type=yaml_settings['vehicles'][1]['gps_type']),
    ]
    detection_data = [
        DetectionData(det_traj_file, plot_name=yaml_settings['detections'][0]['name'],detection_type=yaml_settings['detections'][0]['detection_type'], plot_flag=yaml_settings['detections'][0]['plot_result']),
        DetectionData(det_traj_file2, plot_name=yaml_settings['detections'][1]['name'],detection_type=yaml_settings['detections'][1]['detection_type'], plot_flag=yaml_settings['detections'][1]['plot_result']),
        DetectionData(det_traj_file3, plot_name=yaml_settings['detections'][2]['name'],detection_type=yaml_settings['detections'][2]['detection_type'], plot_flag=yaml_settings['detections'][2]['plot_result']),
        DetectionData(det_traj_file4, plot_name=yaml_settings['detections'][3]['name'],detection_type=yaml_settings['detections'][3]['detection_type'], plot_flag=yaml_settings['detections'][3]['plot_result']),
    ]
    
    ped_data=[PedData(det_ped_file, plot_name='Pedestrian',gps_type='ublox'),
    ]

    plotter = Plotter()

    for d in vehicle_data:
        if d.exist == True:
            plotter.plot_vehicle_data(d)
    for d in ped_data:
        if d.exist == True:
            plotter.plot_ped_data(d)
    for d in detection_data:
        if d.exist == True and d.plot_flag == True:
            plotter.plot_detection(d)
    plotter.fig.show()