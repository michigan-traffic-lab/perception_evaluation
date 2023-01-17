import plotly.graph_objects as go
import collections
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
from visualizer_legacy import CAVData, DetectionData, Plotter

from evaluator import Evaluator

RADIUS = 50
GROUND_TRUTH_FREQUENCY=50 #50hz
GROUND_TRUTH_FREQUENCY2=50 #20hz avalon
DET_FREQUENCY=2.5 #10hz Derq system  Conti
ALPHA=GROUND_TRUTH_FREQUENCY/DET_FREQUENCY
ALPHA2=GROUND_TRUTH_FREQUENCY2/DET_FREQUENCY
DISTANCE_THRESHOLD=1.5 #1.5 meters
TIME_THRESHOLD={'Msight':0.5,'Conti':-3,'Derq':-1} #0.5 for Msight  -3 for conti  -1 for Derq
LATENCY=1.8 #change to real value when calculate longitudinal errors

parser = argparse.ArgumentParser()
parser.add_argument('--trial-name', default='8')
parser.add_argument('--experiment-name', default='11-15-2022')
parser.add_argument('--config-file', '-c', help="configuration file path", default='experiment_settings.yml')
args = parser.parse_args()

# det_traj_file = './pred_ne_v2_filtered_8.csv'
det_traj_file = './data/pred_filtered_4gs_v2_4.csv'
cav_1_rtk_file = './data/gt_4.csv'


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
    CENTER_COR=[42.229394, -83.739015]# intersection center point

    # dx, dy = 111139 * (np.array([data['lat'],data['lon']])-np.array(CENTER_COR))
    # d = (dx ** 2 + dy ** 2) ** 0.5
    d = compute_real_meter_distance_from_latlon(CENTER_COR[0], CENTER_COR[1], data['lat'], data['lon'])
    if d>RADIUS:
        out_of_range=True
    else:
        out_of_range=False
    return out_of_range

def min_dist(detection_data, i, vehicle_data):#lat_p, lon_p, lats_gt, lons_gt
    lat_p=detection_data.lats[i] 
    lon_p=detection_data.longs[i]
    lats_gt=vehicle_data.lats 
    lons_gt=vehicle_data.longs
    
    if detection_data.detection_type=='Msight':
        t_detection=float(datetime.datetime.strptime(detection_data.datetime[i], '%Y-%m-%d %H-%M-%S-%f').timestamp()*1000000000)
    else:
        t_detection=float(datetime.datetime.strptime(detection_data.datetime[i], '%Y-%m-%d %H:%M:%S.%f').timestamp()*1000000000)
    
    t_groundtruth=[float(i) for i in vehicle_data.datetime]

    time_diff=abs(t_detection-np.array(t_groundtruth))/1000000000

    dx, dy = 111139 * (lat_p - np.array(lats_gt)), 111139 * \
        (lon_p - np.array(lons_gt))
    d = (dx ** 2 + dy ** 2) ** 0.5
    sorted_d=np.sort(d)
    for item in sorted_d:
        index=np.where(d == item)
        if time_diff[index[0][0]]<10:
            d_min=item
            d_index=index[0][0]
            break
    #d_index = min((range(len(d))), key=d.__getitem__)
    return d_min, d_index

def accuracy(vehicle_data, detection_data,plotter):
    if len(vehicle_data) == 2:
        cav1=vehicle_data[0]
        cav2=vehicle_data[1]
        cav1_remap=[0]*len(cav1.lats)
        cav2_remap=[0]*len(cav2.lats)
    else:
        cav1=vehicle_data[0]
        cav1_remap=[0]*len(cav1.lats)
        cav2=[]
        cav2_remap=[]

    veh1_obj_ids=[]
    veh2_obj_ids=[]
    if len(vehicle_data) == 2:
        for i in range(len(detection_data.lats)):
            d, d_index = min_dist(detection_data, i, vehicle_data[0])
            d1, d1_index = min_dist(detection_data, i, vehicle_data[1])
            if detection_data.detection_type=='Msight':
                df_time_to_veh1=(float(datetime.datetime.strptime(detection_data.datetime[i], '%Y-%m-%d %H-%M-%S-%f').timestamp()*1000000000)
                    - float(vehicle_data[0].datetime[d_index]))/1000000000
                df_time_to_veh2=(float(datetime.datetime.strptime(detection_data.datetime[i], '%Y-%m-%d %H-%M-%S-%f').timestamp()*1000000000)
                    - float(vehicle_data[1].datetime[d1_index]))/1000000000
            else:
                df_time_to_veh1=(float(datetime.datetime.strptime(detection_data.datetime[i], '%Y-%m-%d %H:%M:%S.%f').timestamp()*1000000000)
                    - float(vehicle_data[0].datetime[d_index]))/1000000000
                df_time_to_veh2=(float(datetime.datetime.strptime(detection_data.datetime[i], '%Y-%m-%d %H:%M:%S.%f').timestamp()*1000000000)
                    - float(vehicle_data[1].datetime[d1_index]))/1000000000

            if veh1_obj_ids.count(detection_data.obj_ids[i]):
                cav1_remap[d_index]=[d,detection_data.lats[i], detection_data.longs[i],df_time_to_veh1,detection_data.obj_ids[i]]
            elif veh2_obj_ids.count(detection_data.obj_ids[i]):
                cav2_remap[d1_index]=[d1,detection_data.lats[i], detection_data.longs[i],df_time_to_veh2,detection_data.obj_ids[i]]
            else:
                if d<3 and d1<3:
                    if df_time_to_veh2<TIME_THRESHOLD[detection_data.detection_type]:
                        cav1_remap[d_index]=[d,detection_data.lats[i], detection_data.longs[i],df_time_to_veh1,detection_data.obj_ids[i]]
                        veh1_obj_ids.append(detection_data.obj_ids[i])
                    elif df_time_to_veh2>=TIME_THRESHOLD[detection_data.detection_type]:      
                        cav2_remap[d1_index]=[d1,detection_data.lats[i], detection_data.longs[i],df_time_to_veh2,detection_data.obj_ids[i]]
                        veh2_obj_ids.append(detection_data.obj_ids[i])
                elif d<d1:
                    cav1_remap[d_index]=[d,detection_data.lats[i], detection_data.longs[i],df_time_to_veh1,detection_data.obj_ids[i]]
                    veh1_obj_ids.append(detection_data.obj_ids[i])
                else:
                    cav2_remap[d1_index]=[d1,detection_data.lats[i], detection_data.longs[i],df_time_to_veh2,detection_data.obj_ids[i]]
                    veh2_obj_ids.append(detection_data.obj_ids[i])
    else:
        for i in range(len(detection_data.lats)):
            #d, d_index = min_dist(detection_data.lats[i], detection_data.longs[i], vehicle_data[0].lats, vehicle_data[0].longs)
            d, d_index = min_dist(detection_data, i, vehicle_data[0])

            if detection_data.detection_type=='Msight':
                df_time_to_veh1=(float(datetime.datetime.strptime(detection_data.datetime[i], '%Y-%m-%d %H-%M-%S-%f').timestamp()*1000000000)
                - float(vehicle_data[0].datetime[d_index]))/1000000000
            else:
                df_time_to_veh1=(float(datetime.datetime.strptime(detection_data.datetime[i], '%Y-%m-%d %H:%M:%S.%f').timestamp()*1000000000)
                 - float(vehicle_data[0].datetime[d_index]))/1000000000
            if cav1_remap[d_index]==0:
                cav1_remap[d_index]=[d,detection_data.lats[i], detection_data.longs[i],df_time_to_veh1,detection_data.obj_ids[i]]
            elif d<cav1_remap[d_index][0]:
                cav1_remap[d_index]=[d,detection_data.lats[i], detection_data.longs[i],df_time_to_veh1,detection_data.obj_ids[i]]
            ##Intersection area may has issue right now. Add time check later to resolve this issue

    #false negative rate
    det1_list=[i for i in cav1_remap if i!=0]
    number_of_detected_v1=0
    for item in cav1_remap:
        if item!= 0:
            if item[0]<DISTANCE_THRESHOLD:
                number_of_detected_v1+=1

    
    FN_result=round((len(cav1_remap)-(ALPHA*len(det1_list)))/len(cav1_remap)*100,2)
    
    print('-------------------------False negative rate (%)------------------------------')
    print('-------------Percents of False negative in ground truth-----------------------')
    print(f'CAV 1: {FN_result}')
    print(f'Data frequency: {round(len(det1_list)/len(cav1_remap)*GROUND_TRUTH_FREQUENCY,2)}')
    
    
    if len(vehicle_data) == 2:
        det2_list=[i for i in cav2_remap if i!=0]
        number_of_detected_v2=0
        for i in cav2_remap:
            if i != 0:
                if i[0]<DISTANCE_THRESHOLD:
                    number_of_detected_v2+=1
        fn2=round((len(cav2_remap)-(ALPHA2*len(det2_list)))/len(cav2_remap)*100,2)
        print(f'CAV 2: {fn2}')
        print(f'Data frequency: {round(len(det2_list)/len(cav2_remap)*GROUND_TRUTH_FREQUENCY2,2)}')

    FP_count_v1=0
    #false postive rate
    for item in cav1_remap:
        if item!= 0:
            if item[0]>=DISTANCE_THRESHOLD:
                FP_count_v1+=1
    
    FP_result=round((FP_count_v1/len(det1_list))*100,2)

    print('-------------------------False positive rate (%)------------------------------')
    print('-------------Percents of False Positive in detection result-------------------')
    print(f'CAV 1: {FP_result}')
    if len(vehicle_data) == 2:
        FP_count_v2=0
        for i in cav2_remap:
            if i != 0 and i[0]>=DISTANCE_THRESHOLD:
                FP_count_v2+=1
        fp2=round((FP_count_v2/len(det2_list))*100,2)
        print(f'CAV 2: {fp2}')

    ##offset
    cul=[]
    for item in cav1_remap:
        if item!= 0:
            if item[0]<DISTANCE_THRESHOLD:
                cul.append(item[0])
            
    cul1=round(np.mean(cul),2)
    var1=round(np.var(cul),2)

    print('-------------------------Lateral positional error (m)-------------------------')
    print(f'CAV 1: mean {cul1}, variance {var1}')
    if len(vehicle_data) == 2:
        count2=0
        cul2=[]
        for i in cav2_remap:
            if i != 0 and i[0]<DISTANCE_THRESHOLD:
                count2+=1
                cul2.append(i[0])
        offset2=round(np.mean(cul2),2)
        var2=round(np.var(cul2),2)
        print(f'CAV 2: mean {offset2}, variance {var2}')
    
    # plotter.plot_matching(vehicle_data[0],cav1_remap,'blue')
    # plotter.plot_matching(vehicle_data[1],cav2_remap,'red')

    no_zero_cav1_id = [i[4] for i in cav1_remap if i != 0]
    cav1_ID_switch_time=len(set(no_zero_cav1_id))-1
    cav1_counter=collections.Counter(no_zero_cav1_id)

    if len(vehicle_data) == 2:
        no_zero_cav2_id = [i[4] for i in cav2_remap if i != 0]
        cav2_ID_switch_time=len(set(no_zero_cav2_id))-1
        cav2_counter=collections.Counter(no_zero_cav2_id)
    
    print('-------------------------ID consistency---------------------------------------')
    print(f'CAV 1 ID switch time: {cav1_ID_switch_time}')
    print(f'CAV 1 longest ID: {max(cav1_counter)} percents: {round(max(cav1_counter.values())/len(no_zero_cav1_id),2)}')
    if len(vehicle_data) == 2:
        print(f'CAV 2 ID switch time: {cav2_ID_switch_time}')
        print(f'CAV 2 longest ID: {max(cav2_counter)} percents: {round(max(cav2_counter.values())/len(no_zero_cav2_id),2)}')

    print('-------------------------MOTA---------------------------------------')
    FN_points_v1=len(cav1_remap)/ALPHA-number_of_detected_v1
    FP_points_v1=FP_count_v1
    
    MOTA1=1-((FN_points_v1+FP_points_v1+cav1_ID_switch_time)/((len(cav1_remap)/ALPHA)+FP_points_v1))

    print(f'CAV 1 MOTA: {round(MOTA1,2)}')
    if len(vehicle_data) == 2:
        FN_points_v2=len(cav2_remap)/ALPHA2-number_of_detected_v2
        FP_points_v2=FP_count_v2
        MOTA2=1-((FN_points_v2+FP_points_v2+cav2_ID_switch_time)/((len(cav2_remap)/ALPHA2)+FP_points_v2))
        print(f'CAV 2 MOTA: {round(MOTA2,2)}')

    print('-------------------------One-way latency (s)----------------------------------')
    res_lat_one_way=[]
    for item in cav1_remap:
        if item!= 0:
            res_lat_one_way.append(item[3])
    print(f'One-way latency: {round(np.mean(res_lat_one_way),3)}')
    #note one-way latency is not final latency, do the average with the other direction's latency to offset the longitudinal error.
    #once we get system latency, then uncomment the following scripts to calculate longitudinal error.
    
    #######start from here
    cav1_remap_refer_time=[0]*len(cav1.lats)
    
    for i in range(len(detection_data.lats)):
        if detection_data.detection_type=='Msight':
            a=float(datetime.datetime.strptime(detection_data.datetime[i], '%Y-%m-%d %H-%M-%S-%f').timestamp()*1000000000)
        else:
            a=float(datetime.datetime.strptime(detection_data.datetime[i], '%Y-%m-%d %H:%M:%S.%f').timestamp()*1000000000)
        b=LATENCY*1000000000
        t_diff= [(a-b-float(item))/1000000000 for item in vehicle_data[0].datetime]
        t_min=min(np.abs(t_diff))
        t_min_index=min((range(len(np.abs(t_diff)))), key=np.abs(t_diff).__getitem__)
        if t_min<1/50:
            if cav1_remap_refer_time[t_min_index]==0:
                cav1_remap_refer_time[t_min_index]=[t_min,detection_data.lats[i], detection_data.longs[i]]
        
    
    plotter.plot_matching(vehicle_data[0],cav1_remap_refer_time,'blue')
    long_error=[]
    for i in range(len(cav1_remap_refer_time)):
        for j in range(len(cav1_remap)):
            if cav1_remap_refer_time[i]!=0 and cav1_remap[j]!=0:
                if cav1_remap[j][1]==cav1_remap_refer_time[i][1] and cav1_remap[j][2]==cav1_remap_refer_time[i][2]:
                    d=distance.geodesic([vehicle_data[0].lats[i],vehicle_data[0].longs[i]], [vehicle_data[0].lats[j],vehicle_data[0].longs[j]]).m
                    long_error.append(d)
    print('-------------------------Longitudinal error (m)-------------------------------')
    print(f'CAV 1: mean {round(np.mean(long_error),2)}, variance {round(np.var(long_error),2)}')
    #########end at here

if __name__  == '__main__':
    vehicle_data = [
        # CAVData([cav_1_rtk_file, cav_1_speed_file], plot_name=yaml_settings['vehicles'][0]['name'],gps_type='oxford'),
        # CAVData(cav_2_rtk_file, plot_name=yaml_settings['vehicles'][1]['name'],gps_type='oxford-new'),
        CAVData(cav_1_rtk_file, plot_name='CAV 1 trajectory recorded by RTK', gps_type='oxford-new'),
    ]
    detection_data = [
        DetectionData(det_traj_file, plot_name='detection system 1', detection_type='Msight'),
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
    
    ## evaluate accuracy
    evaluator = Evaluator(1.5, detection_data[0], vehicle_data[0], latency=LATENCY, 
                          det_freq=DET_FREQUENCY, gt_freq=GROUND_TRUTH_FREQUENCY)
    evaluator.evaluate()

    # plotter.fig.show()
    