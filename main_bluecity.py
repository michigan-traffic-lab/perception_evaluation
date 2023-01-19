import argparse
from utils_mcity_legacy import CAVData, DetectionData
from evaluator import Evaluator
import numpy as np

from utils_ped import PedRTKData

GROUND_TRUTH_FREQUENCY_VEH1 = 50  # 50hz
GROUND_TRUTH_FREQUENCY_VEH2 = 100  # 50hz
DET_FREQUENCY = 10  # 10hz Derq system  Conti
DISTANCE_THRESHOLD = 1.5  # 1.5 meters
VEH_LATENCY = 0.0807  # change to real value when calculate longitudinal errors
PED_LATENCY = None
CENTER_COR = [42.300947, -83.698649]


def veh_evaluation():
    global VEH_LATENCY
    if VEH_LATENCY is None:
        print('Estimating system vehicle detection latency...')
        for trip_id in range(1, 3):
            det_traj_file = f'data/bluecity_20230105/detection_{trip_id}.json'
            cav_1_rtk_file = f'data/mcity-perception-data/TestingResult/01-05-2023/{trip_id}/gps_fix_{trip_id}.txt'
            cav_1_speed_file = f'data/mcity-perception-data/TestingResult/01-05-2023/{trip_id}/gps_vel_{trip_id}.txt'
            vehicle_data = CAVData([cav_1_rtk_file, cav_1_speed_file], plot_name='CAV 1 trajectory recorded by RTK',
                                   gps_type='oxford')
            detection_data = DetectionData(det_traj_file, plot_name='detection system 1', detection_type='Bluecity')

            latency_list = []
            print(f'++++++++++++++++++ For trip {trip_id} +++++++++++++++++++')
            ## evaluate accuracy
            evaluator = Evaluator('veh', 1.5, detection_data, vehicle_data, det_freq=DET_FREQUENCY,
                                  gt_freq=GROUND_TRUTH_FREQUENCY_VEH1, center_latlng=CENTER_COR, roi_radius=50)
            latency_list.append(evaluator.compute_latency())
            VEH_LATENCY = np.mean(np.array(latency_list))
    print(f'Estimated system vehicle detection latency: {VEH_LATENCY} s')
    # evaluate perception system on single-vehicle trips
    print('Evaluating system perception performance on single-vehicle scenerios...')
    fp_list = []
    fn_list = []
    det_freq_list = []
    lat_error_list = []
    lon_error_list = []
    id_switch_list = []
    id_consistency_list = []
    mota_list = []
    for trip_id in range(1, 13):
        det_traj_file = f'data/bluecity_20230105/detection_{trip_id}.json'
        cav_1_rtk_file = f'data/mcity-perception-data/TestingResult/01-05-2023/{trip_id}/gps_fix_{trip_id}.txt'
        cav_1_speed_file = f'data/mcity-perception-data/TestingResult/01-05-2023/{trip_id}/gps_vel_{trip_id}.txt'
        vehicle_data = CAVData([cav_1_rtk_file, cav_1_speed_file], plot_name='CAV 1 trajectory recorded by RTK',
                               gps_type='oxford')
        detection_data = DetectionData(det_traj_file, plot_name='detection system 1', detection_type='Bluecity')

        print(f'++++++++++++++++++ For trip {trip_id} +++++++++++++++++++')
        ## evaluate accuracy
        evaluator = Evaluator('veh', 1.5, detection_data, vehicle_data, latency=VEH_LATENCY,
                              det_freq=DET_FREQUENCY, gt_freq=GROUND_TRUTH_FREQUENCY_VEH1, center_latlng=CENTER_COR,
                              roi_radius=50)
        fp, fn, det_freq, lat_error, lon_error, id_switch, id_consistency, mota = evaluator.evaluate(visualize=False)
        fp_list.append(fp)
        fn_list.append(fn)
        det_freq_list.append(det_freq)
        lat_error_list.append(lat_error)
        lon_error_list.append(lon_error)
        id_switch_list.append(id_switch)
        id_consistency_list.append(id_consistency)
        mota_list.append(mota)
    # evaluate perception system on two-vehicle trips
    print('Evaluating system perception performance on two-vehicle scenerios...')
    for trip_id in range(13, 16):
        for veh in ['veh1', 'veh2']:
            det_traj_file = f'data/bluecity_20230105/detection_{trip_id}_{veh}.json'
            cav_1_rtk_file = f'data/mcity-perception-data/TestingResult/01-05-2023/{trip_id}/gps_fix_{veh}_{trip_id}.txt'
            cav_1_speed_file = f'data/mcity-perception-data/TestingResult/01-05-2023/{trip_id}/gps_vel_{veh}_{trip_id}.txt'
            vehicle_data = CAVData([cav_1_rtk_file, cav_1_speed_file], plot_name='CAV 1 trajectory recorded by RTK',
                                   gps_type='oxford')
            detection_data = DetectionData(det_traj_file, plot_name='detection system 1', detection_type='Bluecity')

            print(f'++++++++++++++++++ For trip {trip_id} {veh} +++++++++++++++++++')
            ## evaluate accuracy
            evaluator = Evaluator('veh', 1.5, detection_data, vehicle_data, latency=VEH_LATENCY,
                                  det_freq=DET_FREQUENCY,
                                  gt_freq=GROUND_TRUTH_FREQUENCY_VEH1 if veh == 'veh1' else GROUND_TRUTH_FREQUENCY_VEH2,
                                  center_latlng=CENTER_COR, roi_radius=50)
            fp, fn, det_freq, lat_error, lon_error, id_switch, id_consistency, mota = evaluator.evaluate(visualize=False)
            fp_list.append(fp)
            fn_list.append(fn)
            det_freq_list.append(det_freq)
            lat_error_list.append(lat_error)
            lon_error_list.append(lon_error)
            id_switch_list.append(id_switch)
            id_consistency_list.append(id_consistency)
            mota_list.append(mota)
    # summarizing results
    mean_fp = np.mean(np.array(fp_list))
    mean_fn = np.mean(np.array(fn_list))
    mean_det_freq = np.mean(np.array(det_freq_list))
    mean_lat_error = np.mean(np.array(lat_error_list))
    mean_lon_error = np.mean(np.array(lon_error_list))
    mean_id_switch = np.mean(np.array(id_switch_list))
    mean_id_consistency = np.mean(np.array(id_consistency_list))
    mean_mota = np.mean(np.array(mota_list))
    print('Overall performance: ')
    print('-------------------------------------------------------')
    print('---------- Overall Evaluation Results -----------------')
    print('-------------------------------------------------------')
    print(f'False Positive rate\t\t: {mean_fp:.4f}')
    print(f'False Negative rate\t\t: {mean_fn:.4f}')
    print(f'Actual detection frequency\t: {mean_det_freq:.4f}')
    print(f'Lateral error\t\t\t: {mean_lat_error:.4f}')
    print(f'Longitudinal error\t\t: {mean_lon_error:.4f}')
    print(f'System Latency\t\t\t: {VEH_LATENCY:.4f}')
    print(f'ID switch\t\t\t: {mean_id_switch:.4f}')
    print(f'ID consistency\t\t\t: {mean_id_consistency:.4f}')
    print(f'MOTA\t\t\t\t: {mean_mota:.4f}')
    print('-------------------------------------------------------')
    print('---------- End of Evaluation Results ------------------')
    print('-------------------------------------------------------')


def ped_evaluation(data_dir, experiment_date_mdy):
    global PED_LATENCY
    if PED_LATENCY is None:
        print('Estimating system pedestrian detection latency...')
        # TODO: ADD EXPERIMENT TO PED DETECTION LATENCY
    print(f'Estimated system pedestrian detection latency: {PED_LATENCY} s')
    # evaluate perception system on single-vehicle trips
    print('Evaluating system perception performance on ped...')

    m, d, y = experiment_date_mdy.split('-')
    local_date_str = '-'.join([y, m, d])

    fp_list = []
    fn_list = []
    det_freq_list = []
    lat_error_list = []
    lon_error_list = []
    id_switch_list = []
    id_consistency_list = []
    mota_list = []
    for trip_id in range(11, 12):
        ped_det_file_path = f'{data_dir}/{experiment_date_mdy}/splitted_output/detection_{trip_id}-1.json'
        ped_rtk_file_path = f'{data_dir}/{experiment_date_mdy}/{trip_id}/ped_{trip_id}-1.txt'
        ped_rtk_data = PedRTKData(ped_rtk_file_path, plot_name='PED trajectory by RTK', date_str=local_date_str)
        detection_data = DetectionData(ped_det_file_path,
                                       plot_name='PED trajectory by detection system',
                                       detection_type='Bluecity')

        print(f'++++++++++++++++++ For trip {trip_id} +++++++++++++++++++')
        ## evaluate accuracy
        evaluator = Evaluator('ped', 1.5, detection_data, ped_rtk_data, latency=PED_LATENCY,
                              det_freq=DET_FREQUENCY, gt_freq=GROUND_TRUTH_FREQUENCY_VEH1, center_latlng=CENTER_COR,
                              roi_radius=50)
        fp, fn, det_freq, lat_error, lon_error, id_switch, id_consistency, mota = evaluator.evaluate(visualize=True)
        fp_list.append(fp)
        fn_list.append(fn)
        det_freq_list.append(det_freq)
        lat_error_list.append(lat_error)
        lon_error_list.append(lon_error)
        id_switch_list.append(id_switch)
        id_consistency_list.append(id_consistency)
        mota_list.append(mota)

    # summarizing results
    mean_fp = np.mean(np.array(fp_list))
    mean_fn = np.mean(np.array(fn_list))
    mean_det_freq = np.mean(np.array(det_freq_list))
    mean_lat_error = np.mean(np.array(lat_error_list))
    mean_lon_error = np.mean(np.array(lon_error_list))
    mean_id_switch = np.mean(np.array(id_switch_list))
    mean_id_consistency = np.mean(np.array(id_consistency_list))
    mean_mota = np.mean(np.array(mota_list))
    print('Overall performance: ')
    print('-------------------------------------------------------')
    print('---------- Overall Evaluation Results -----------------')
    print('-------------------------------------------------------')
    print(f'False Positive rate\t\t: {mean_fp:.4f}')
    print(f'False Negative rate\t\t: {mean_fn:.4f}')
    print(f'Actual detection frequency\t: {mean_det_freq:.4f}')
    print(f'Lateral error\t\t\t: {mean_lat_error:.4f}')
    print(f'Longitudinal error\t\t: {mean_lon_error:.4f}')
    print(f'System Latency\t\t\t: {PED_LATENCY:.4f}')
    print(f'ID switch\t\t\t: {mean_id_switch:.4f}')
    print(f'ID consistency\t\t\t: {mean_id_consistency:.4f}')
    print(f'MOTA\t\t\t\t: {mean_mota:.4f}')
    print('-------------------------------------------------------')
    print('---------- End of Evaluation Results ------------------')
    print('-------------------------------------------------------')


if __name__ == '__main__':
    # veh_evaluation()
    experiment_date_mdy = '01-05-2023'
    ped_data_dir = '/home/zihao/Projects/mcity-perception-data/TestingResult'
    ped_evaluation(ped_data_dir, experiment_date_mdy)
