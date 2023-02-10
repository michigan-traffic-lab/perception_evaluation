import argparse
from utils_mcity_legacy import CAVData, DetectionData
from evaluator import Evaluator
import numpy as np
from utils import prepare_data

from utils_ped import PedRTKData

GROUND_TRUTH_FREQUENCY_VEH1 = 100  # 50hz
GROUND_TRUTH_FREQUENCY_VEH2 = 50  # 50hz
GROUND_TRUTH_FREQUENCY_PED = 10  # 50hz
DET_FREQUENCY = 10  # 10hz Derq system  Conti
DISTANCE_THRESHOLD = 1.5  # 1.5 meters
VEH_LATENCY = 0.2058  # change to real value when calculate longitudinal errors
PED_LATENCY = None
CENTER_COR = [42.300947, -83.698649]
EXPERIMENT_DATA_DIR = ''
LATENCY_TRIAL_IDS = [1, 2]
SINGLE_VEH_TRIAL_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
TWO_VEH_TRIAL_IDS = [13, 14, 15]
PED_TRIAL_IDS = [11]

def veh_visualization_temp_two_vehicle(data_dir, experiment_date_mdy, trial_id):
    det_traj_file = f'{data_dir}/{experiment_date_mdy}/{trial_id}/edge_DSRC_BSM_send_{trial_id}.csv'
    cav_1_rtk_file = f'{data_dir}/{experiment_date_mdy}/{trial_id}/gps_fix_veh1_{trial_id}.txt'
    cav_1_speed_file = f'{data_dir}/{experiment_date_mdy}/{trial_id}/gps_vel_veh1_{trial_id}.txt'
    vehicle_data1 = CAVData([cav_1_rtk_file, cav_1_speed_file], plot_name='CAV 1 trajectory recorded by RTK',
                            gps_type='oxford')
    cav_2_rtk_file = f'{data_dir}/{experiment_date_mdy}/{trial_id}/gps_fix_veh2_{trial_id}.txt'
    cav_2_speed_file = f'{data_dir}/{experiment_date_mdy}/{trial_id}/gps_vel_veh2_{trial_id}.txt'
    vehicle_data2 = CAVData([cav_2_rtk_file, cav_2_speed_file], plot_name='CAV 1 trajectory recorded by RTK',
                            gps_type='oxford')
    detection_data = DetectionData(det_traj_file, plot_name='detection system 1', detection_type='Derq')
    dtdp_list, gtdp_list_1 = prepare_data(detection_data, vehicle_data1, source='Derq')
    dtdp_list, gtdp_list_2 = prepare_data(detection_data, vehicle_data2, source='Derq')

    evaluator = Evaluator('Derq', 'veh', 1.5, detection_data, vehicle_data1, latency=0.1, det_freq=DET_FREQUENCY,
                            gt_freq=GROUND_TRUTH_FREQUENCY_VEH1, center_latlng=CENTER_COR, roi_radius=50)
    evaluator.visualize_data_temp(dtdp_list, [gtdp_list_1, gtdp_list_2])
    # evaluator.visualize_data()

def veh_evaluation(data_dir, experiment_date_mdy, visualize=False):
    # prepare vehicle trajecoty files
    global VEH_LATENCY
    if VEH_LATENCY is None:
        print('Estimating system vehicle detection latency...')
        latency_list = []
        for trial_id in LATENCY_TRIAL_IDS:
            det_traj_file = f'{data_dir}/{experiment_date_mdy}/{trial_id}/edge_DSRC_BSM_send_{trial_id}.csv'
            cav_1_rtk_file = f'{data_dir}/{experiment_date_mdy}/{trial_id}/gps_fix_{trial_id}.txt'
            cav_1_speed_file = f'{data_dir}/{experiment_date_mdy}/{trial_id}/gps_vel_{trial_id}.txt'
            vehicle_data = CAVData([cav_1_rtk_file, cav_1_speed_file], plot_name='CAV 1 trajectory recorded by RTK',
                                   gps_type='oxford')
            detection_data = DetectionData(det_traj_file, plot_name='detection system 1', detection_type='Derq')

            print(f'++++++++++++++++++ For trip {trial_id} +++++++++++++++++++')
            ## evaluate accuracy
            evaluator = Evaluator('Derq', 'veh', 1.5, detection_data, vehicle_data, latency=None, det_freq=DET_FREQUENCY,
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
    for trial_id in SINGLE_VEH_TRIAL_IDS:
        det_traj_file = f'{data_dir}/{experiment_date_mdy}/{trial_id}/edge_DSRC_BSM_send_{trial_id}.csv'
        cav_1_rtk_file = f'{data_dir}/{experiment_date_mdy}/{trial_id}/gps_fix_{trial_id}.txt'
        cav_1_speed_file = f'{data_dir}/{experiment_date_mdy}/{trial_id}/gps_vel_{trial_id}.txt'
        vehicle_data = CAVData([cav_1_rtk_file, cav_1_speed_file], plot_name='CAV 1 trajectory recorded by RTK',
                               gps_type='oxford')
        detection_data = DetectionData(det_traj_file, plot_name='detection system 1', detection_type='Derq')

        print(f'++++++++++++++++++ For trip {trial_id} +++++++++++++++++++')
        ## evaluate accuracy
        evaluator = Evaluator('Derq', 'veh', 1.5, detection_data, vehicle_data, latency=VEH_LATENCY,
                              det_freq=DET_FREQUENCY, gt_freq=GROUND_TRUTH_FREQUENCY_VEH1, center_latlng=CENTER_COR,
                              roi_radius=50)
        fp, fn, det_freq, lat_error, lon_error, id_switch, id_consistency, mota = evaluator.evaluate(visualize=visualize)
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
    for trial_id in TWO_VEH_TRIAL_IDS:
        for veh in ['veh1', 'veh2']:
            det_traj_file = f'{data_dir}/{experiment_date_mdy}/{trial_id}/edge_DSRC_BSM_send_{trial_id}_{veh}.csv'
            cav_1_rtk_file = f'{data_dir}/{experiment_date_mdy}/{trial_id}/gps_fix_{veh}_{trial_id}.txt'
            cav_1_speed_file = f'{data_dir}/{experiment_date_mdy}/{trial_id}/gps_vel_{veh}_{trial_id}.txt'
            vehicle_data = CAVData([cav_1_rtk_file, cav_1_speed_file], plot_name='CAV 1 trajectory recorded by RTK',
                                   gps_type='oxford')
            detection_data = DetectionData(det_traj_file, plot_name='detection system 1', detection_type='Derq')

            print(f'++++++++++++++++++ For trip {trial_id} {veh} +++++++++++++++++++')
            ## evaluate accuracy
            evaluator = Evaluator('Derq', 'veh', 1.5, detection_data, vehicle_data, latency=VEH_LATENCY,
                                  det_freq=DET_FREQUENCY,
                                  gt_freq=GROUND_TRUTH_FREQUENCY_VEH1 if veh == 'veh1' else GROUND_TRUTH_FREQUENCY_VEH2,
                                  center_latlng=CENTER_COR, roi_radius=50)
            fp, fn, det_freq, lat_error, lon_error, id_switch, id_consistency, mota = evaluator.evaluate(visualize=visualize)
            evaluator.visualize_data()
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


def ped_evaluation(data_dir, experiment_date_mdy, visualize=False):
    global PED_LATENCY
    PED_LATENCY = VEH_LATENCY
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
    for trial_id in PED_TRIAL_IDS:
        ped_det_file_path = f'{data_dir}/{experiment_date_mdy}/detection_results/detection_{trial_id}.json'
        ped_rtk_file_path = f'{data_dir}/{experiment_date_mdy}/{trial_id}/ped_{trial_id}.txt'
        ped_rtk_data = PedRTKData(ped_rtk_file_path, plot_name='PED trajectory by RTK', date_str=local_date_str)
        detection_data = DetectionData(ped_det_file_path,
                                       plot_name='PED trajectory by detection system',
                                       detection_type='Derq')

        print(f'++++++++++++++++++ For trip {trial_id} +++++++++++++++++++')
        ## evaluate accuracy
        evaluator = Evaluator('Derq', 'ped', 1.5, detection_data, ped_rtk_data, latency=PED_LATENCY,
                              det_freq=DET_FREQUENCY, gt_freq=GROUND_TRUTH_FREQUENCY_PED, center_latlng=CENTER_COR,
                              roi_radius=50)
        fp, fn, det_freq, lat_error, lon_error, id_switch, id_consistency, mota = evaluator.evaluate(visualize=visualize)
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
    experiment_date_mdy = '12-19-2022'
    data_dir = 'data/mcity-perception-data/TestingResult'
    veh_visualization_temp_two_vehicle(data_dir, experiment_date_mdy, 15)
    # veh_evaluation(data_dir, experiment_date_mdy, visualize=True)
    # ped_evaluation(data_dir, experiment_date_mdy)
