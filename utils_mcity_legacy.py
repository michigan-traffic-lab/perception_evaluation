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

RADIUS = 50

parser = argparse.ArgumentParser()
parser.add_argument('--trial-name', default='3')
parser.add_argument('--experiment-name', default='11-15-2022')
parser.add_argument('--config-file', '-c', help="configuration file path", default='experiment_settings.yml')
args = parser.parse_args()

# with open(args.config_file, 'r') as f:
#     yaml_settings = yaml.safe_load(f)

# if yaml_settings['data-dir'] is None:
#     HomeDir = Path.home()
#     if yaml_settings['system'] == 'linux':
#         cav_1_rtk_file = HomeDir / 'Archive/TestingResult' / args.experiment_name / str(args.trial_name) / f'gps_fix_{args.trial_name}.txt'
#         cav_1_speed_file = HomeDir / 'Archive/TestingResult' / args.experiment_name / str(args.trial_name) / f'gps_vel_{args.trial_name}.txt'
#         cav_2_rtk_file = HomeDir / 'Archive/TestingResult' / args.experiment_name / str(args.trial_name) / f'gps_veh2_{args.trial_name}.gps'
#         det_traj_file = HomeDir / 'Archive/TestingResult' / args.experiment_name / str(args.trial_name) / f'edge_DSRC_BSM_send_{args.name}.csv'
#     else:
#         cav_1_rtk_file = str(HomeDir) + str(pathlib.PureWindowsPath('/Documents/GitHub/Archive/TestingResult/' + args.experiment_name + '/' + str(args.trial_name) + '/gps_fix_' + str(
#             args.trial_name) + '.txt'))
#         cav_1_speed_file = str(HomeDir) + str(pathlib.PureWindowsPath('/Documents/GitHub//Archive/TestingResult/' + args.experiment_name + '/' + str(args.trial_name) + '/gps_vel_' + str(
#             args.trial_name) + '.txt'))
#         #cav_2_rtk_file = str(HomeDir) + str(pathlib.PureWindowsPath('/Documents/GitHub//Archive/TestingResult/' + args.experiment_name + '/' + str(args.trial_name) + '/gps_veh2_' + str(
#         #    args.trial_name) + '.gps'))
#         cav_2_rtk_file = str(HomeDir) + str(pathlib.PureWindowsPath('/Documents/GitHub//Archive/TestingResult/' + args.experiment_name + '/' + str(args.trial_name) + '/gps_veh2_' + str(
#             args.trial_name) + '.csv'))
#         det_traj_file = str(HomeDir) + str(pathlib.PureWindowsPath('/Documents/GitHub//Archive/TestingResult/' + args.experiment_name + '/' + str(args.trial_name) + '/edge_DSRC_BSM_send_' + str(
#             args.trial_name) + '.csv'))
#         det_traj_file2 = str(HomeDir) + str(pathlib.PureWindowsPath('/Documents/GitHub//Archive/TestingResult/' + args.experiment_name + '/' + 'Mcity_Tests/Exports/Test' + str(
#             args.trial_name) + '_SDSMTx.csv'))
#         det_traj_file3 = str(HomeDir) + str(pathlib.PureWindowsPath('/Documents/GitHub//Archive/TestingResult/' + args.experiment_name + '/' + 'Msight/Test' + str(
#             args.trial_name) + '_MsightObjectList.csv'))
# else:
#     cav_1_rtk_file = Path(yaml_settings['data-dir'])  / args.name / f'gps_fix_{args.name}.txt'
#     det_traj_file = Path(yaml_settings['data-dir'])  / args.name / f'DSRC_BSM_server_receive_{args.name}.csv'
#     cav_2_rtk_file = Path(yaml_settings['data-dir'])  / args.name / f'gps_veh2_{args.name}.gps'
#     cav_1_speed_file = Path(yaml_settings['data-dir'])  / args.name / f'gps_vel_{args.name}.gps'

det_traj_file = './pred_filtered_4gs_v2_2.csv'
cav_1_rtk_file = './gt_2.csv'

class ExperimentData:
    def __init__(self):
        self.longs = []
        self.lats = []
        self.datetime = []
        self.texts = []
        self.speeds = []
        self.exist = True
        
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
    def __init__(self,data_files,plot_name, detection_type):
        super().__init__()
        self.detection_type=detection_type
        self.plot_name=plot_name
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
        elif self.detection_type == 'Msight':
            with open(self.files) as det_traj:
                for row in csv.DictReader(det_traj):
                    self.add_detection_data_Msight(row)

    def add_detection_data(self, row):
        lat=float(row['lat'])/10000000
        long=float(row['lon'])/10000000
        out_of_range=filter_distance_to_center({'lat':lat,'lon':long})

        if not out_of_range:
            self.datetime.append(float(datetime.datetime.strptime(row['UTC Timestamp'], '%Y-%m-%d %H:%M:%S.%f').timestamp() * 1000000000))
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
                    #lat=(RSU_ref['lat']*111111+float(row[f'pb.SmartCities.sensorDataSharingMessage.objects.objectData.{obj}.detObjCommon.pos.offsetY']))/111111
                    #long=(RSU_ref['lon']*111111+float(row[f'pb.SmartCities.sensorDataSharingMessage.objects.objectData.{obj}.detObjCommon.pos.offsetX']))/111111
                    #lat=(RSU_ref['lat']*111111+float(row[f'pb.SmartCities.sensorDataSharingMessage.objects.objectData.{obj}.detObjCommon.pos.offsetY']))/111111
                    #long=(RSU_ref['lon']*111111*np.cos(RSU_ref['lat'] * np.pi/180)+float(row[f'pb.SmartCities.sensorDataSharingMessage.objects.objectData.{obj}.detObjCommon.pos.offsetX']))/(111111*np.cos(RSU_ref['lat'] * np.pi/180))
                    lat=RSU_ref['lat'] + (float(row[f'pb.SmartCities.sensorDataSharingMessage.objects.objectData.{obj}.detObjCommon.pos.offsetY']) / r_earth) * (180 / np.pi)
                    long=RSU_ref['lon'] + (float(row[f'pb.SmartCities.sensorDataSharingMessage.objects.objectData.{obj}.detObjCommon.pos.offsetX']) / r_earth) * (180 / np.pi) / np.cos(RSU_ref['lat'] * np.pi/180)
                except:
                    lat, long=42.301004, -83.697832#if either lat or long is missing, provide a lat lon in Mcity but away from the ROI
                    out_of_range=True
                #if not out_of_range:
                    #print(row[f'pb.SmartCities.sensorDataSharingMessage.objects.objectData.{obj}.detObjCommon.speedConfidence'])
                out_of_range=filter_distance_to_center({'lat':lat,'lon':long})
                #out_of_range=False
                if not out_of_range:
                    datetime_mill = float(row['sender_timestamp']) / 1000000
                    raise NotImplementedError
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
            self.datetime.append(float(datetime.strptime(row['secMarkDecoded'], '%Y-%m-%d %H-%M-%S-%f').timestamp() * 1000000000))
            self.longs.append(long)
            self.lats.append(lat)
            if row['Speed']=='':
                self.speeds.append(0)
            else:
                self.speeds.append(float(row['Speed']))
            self.obj_ids.append(row['Vehicle ID'])
            self.msg_count.append(row['msgCountDecoded'])
            self.texts.append("UTC Time from GPS: " + row['secMarkDecoded'] + 'Speed from CI: ' + str(round(self.speeds[-1], 3)) + ' m/s')

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
                        if data.obj_type[0]=='1':
                            det_text_with_id.append("UTC time when detection result is received by edge: " + str(data.datetime[i]) + '' +
                                                '<br>' + 'CAV Speed from Conti: ' + str(round(det_speed_with_id[-1], 3)) + ' m/s')
                        elif data.obj_type[0]=='2':
                            det_text_with_id.append("UTC time when detection result is received by edge: " + str(data.datetime[i]) + '' +
                                                '<br>' + 'Ped Speed from Conti: ' + str(round(det_speed_with_id[-1], 3)) + ' m/s')
                        elif data.obj_type[0]=='0':
                            det_text_with_id.append("UTC time when detection result is received by edge: " + str(data.datetime[i]) + '' +
                                                '<br>' + 'Unknown Speed from Conti: ' + str(round(det_speed_with_id[-1], 3)) + ' m/s')
                        else:
                            det_text_with_id.append("UTC time when detection result is received by edge: " + str(data.datetime[i]) + '' +
                                                '<br>' + 'Undefined Speed from Conti: ' + str(round(det_speed_with_id[-1], 3)) + ' m/s')
                    if data.detection_type=='Msight':
                        det_text_with_id.append("UTC time when detection result is received by edge: " + str(data.datetime[i]) + '' +
                                            '<br>' + 'Speed from Msight: ' + str(round(det_speed_with_id[-1], 3)) + ' m/s' + 'ID is: '+ str(id))
            if len(data.obj_type) !=0 and len(set(det_obj_type)) != 1:
                print('object type change error')

            
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
                
        self.fig.update_layout(
            mapbox={
                'accesstoken': 'pk.eyJ1Ijoiemh1aGoiLCJhIjoiY2ttdGF2bDd1MHEwbjJybzVxMTduOHVjbSJ9.wT_hzXAry0m33OrdaeTzRA',
                'style': "satellite-streets", 'center': {'lat': (np.mean(data.lats)), 'lon': (np.mean(data.longs))},
                'zoom': 20},
            showlegend=True, title_text='CAV Trajectories Recorded by RTK and Detected by CI')

def compute_real_meter_distance_from_latlon(lat1, lon1, lat2, lon2):  # generally used geo measurement function
    import math
    PI = 3.14159
    if np.isnan(lat1) or np.isnan(lat2) or np.isnan(lon1) or np.isnan(lon2):
        return 10000
    if np.isinf(lat1) or np.isinf(lat2) or np.isinf(lon1) or np.isinf(lon2):
        return 10000
    R = 6378.137    # Radius of earth in KM
    dLat = lat2 * PI / 180.0 - lat1 * PI / 180.0
    dLon = lon2 * PI / 180.0 - lon1 * PI / 180.0
    a = math.sin(dLat/2.0) * math.sin(dLat/2.0) + math.cos(lat1 * PI / 180.0) * math.cos(lat2 * PI / 180.0) * math.sin(dLon/2.0) * math.sin(dLon/2.0)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    d = R * c + 1e-7
    return d * 1000 # meters

# Area of interest filter
def filter_distance_to_center(data):
    CENTER_COR=[42.300947, -83.698649] # intersection center point

    # dx, dy = 111139 * (np.array([data['lat'],data['lon']])-np.array(CENTER_COR))
    # d = (dx ** 2 + dy ** 2) ** 0.5
    d = compute_real_meter_distance_from_latlon(CENTER_COR[0], CENTER_COR[1], data['lat'], data['lon'])
    if d>RADIUS:
        out_of_range=True
    else:
        out_of_range=False
    return out_of_range

if __name__  == '__main__':
    vehicle_data = [
        # CAVData([cav_1_rtk_file, cav_1_speed_file], plot_name=yaml_settings['vehicles'][0]['name'],gps_type='oxford'),
        # CAVData(cav_2_rtk_file, plot_name=yaml_settings['vehicles'][1]['name'],gps_type='oxford-new'),
        CAVData(cav_1_rtk_file, plot_name=yaml_settings['vehicles'][0]['name'],gps_type='oxford-new'),
    ]
    detection_data = [
        DetectionData(det_traj_file, plot_name=yaml_settings['detections'][0]['name'],detection_type='Msight'),
        # DetectionData(det_traj_file2, plot_name=yaml_settings['detections'][1]['name'],detection_type='Conti'),
        #DetectionData(det_traj_file3, plot_name=yaml_settings['detections'][1]['name'],detection_type='Msight'),
    ]
    plotter = Plotter()
    for d in vehicle_data:
        if d.exist == True:
            plotter.plot_vehicle_data(d)
    for d in detection_data:
        if d.exist == True:
            plotter.plot_detection(d)
    plotter.fig.show()