import argparse
from utils_mcity_legacy import CAVData, DetectionData
from evaluator import Evaluator
import numpy as np
from utils import prepare_data, get_detection_file_path
import os
import yaml

from utils_ped import PedRTKData

parser = argparse.ArgumentParser(prog='EvaluationSettingParser',
                                 description='This parser specifies the settings for \
                                              the perception evaluation program')
parser.add_argument('-d', '--data-root',
                    type=str,
                    default='example_data/',
                    help='the root dir of the testing data')
parser.add_argument('-n', '--data-name',
                    default='01-05-2023',
                    type=str,
                    help='the name of the testing, often represented by its testing date')
parser.add_argument('-s', '--system',
                    type=str,
                    default='Bluecity',
                    choices=['Bluecity', 'Derq', 'MSight'],
                    help='The perception system you want to evaluate: Bluecity, Derq, or MSight')
parser.add_argument('-l', '--latency',
                    type=float,
                    default=-1,
                    help='Manually set the system latency, if leave blank, the program will \
                          compute the system latency from trials for latency estimation')
parser.add_argument('-t', '--threshold',
                    type=float,
                    default=1.5,
                    help='The lateral distance threshold for positional evaluations')
parser.add_argument('--visualize', action='store_true')


def veh_evaluation(data_dir, args, cfg):
    # evaluate system latency
    if args.latency == -1:
        print('Estimating system vehicle detection latency...')
        latency_list = []
        for trial_id in cfg['LATENCY_TRIAL_IDS']:
            det_traj_file = get_detection_file_path(
                args.system, data_dir, trial_id)
            cav_1_rtk_file = f'{data_dir}/gts/gps_fix_{trial_id}.txt'
            cav_1_speed_file = f'{data_dir}/gts/gps_vel_{trial_id}.txt'
            vehicle_data = CAVData([cav_1_rtk_file, cav_1_speed_file], plot_name='CAV 1 trajectory recorded by RTK',
                                   gps_type='oxford')
            detection_data = DetectionData(
                det_traj_file, plot_name='detection system 1', detection_type=args.system)
            print(
                f'++++++++++++++++++ For trip {trial_id} +++++++++++++++++++')
            evaluator = Evaluator(args.system, 'veh', 1.5, detection_data, vehicle_data, latency=None, det_freq=cfg['DET_FREQUENCY'],
                                  gt_freq=cfg['GROUND_TRUTH_FREQUENCY_VEH1'], center_latlng=cfg['CENTER_COR'], roi_radius=50)
            latency_list.append(evaluator.compute_latency())
        args.latency = np.mean(np.array(latency_list))
    print(f'Estimated system vehicle detection latency: {args.latency} s')

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
    for trial_id in cfg['SINGLE_VEH_TRIAL_IDS']:
        det_traj_file = get_detection_file_path(
            args.system, data_dir, trial_id)
        cav_1_rtk_file = f'{data_dir}/gts/gps_fix_{trial_id}.txt'
        cav_1_speed_file = f'{data_dir}/gts/gps_vel_{trial_id}.txt'
        vehicle_data = CAVData([cav_1_rtk_file, cav_1_speed_file], plot_name='CAV 1 trajectory recorded by RTK',
                               gps_type='oxford')
        detection_data = DetectionData(
            det_traj_file, plot_name='detection system 1', detection_type=args.system)

        print(f'++++++++++++++++++ For trip {trial_id} +++++++++++++++++++')
        # evaluate accuracy
        evaluator = Evaluator(args.system, 'veh', 1.5, detection_data, vehicle_data, latency=args.latency,
                              det_freq=cfg['DET_FREQUENCY'], gt_freq=cfg['GROUND_TRUTH_FREQUENCY_VEH1'], center_latlng=cfg['CENTER_COR'],
                              roi_radius=50)
        fp, fn, det_freq, lat_error, lon_error, id_switch, id_consistency, mota = evaluator.evaluate(
            visualize=args.visualize)
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
    for trial_id in cfg['TWO_VEH_TRIAL_IDS']:
        for veh in ['veh1', 'veh2']:
            det_traj_file = get_detection_file_path(
                args.system, data_dir, trial_id, veh)
            cav_1_rtk_file = f'{data_dir}/gts/gps_fix_{veh}_{trial_id}.txt'
            cav_1_speed_file = f'{data_dir}/gts/gps_vel_{veh}_{trial_id}.txt'
            vehicle_data = CAVData([cav_1_rtk_file, cav_1_speed_file], plot_name='CAV 1 trajectory recorded by RTK',
                                   gps_type='oxford')
            detection_data = DetectionData(
                det_traj_file, plot_name='detection system 1', detection_type=args.system)

            print(
                f'++++++++++++++++++ For trip {trial_id} {veh} +++++++++++++++++++')
            # evaluate accuracy
            evaluator = Evaluator(args.system, 'veh', 1.5, detection_data, vehicle_data, latency=args.latency,
                                  det_freq=cfg['DET_FREQUENCY'],
                                  gt_freq=cfg['GROUND_TRUTH_FREQUENCY_VEH1'] if veh == 'veh1' else cfg['GROUND_TRUTH_FREQUENCY_VEH2'],
                                  center_latlng=cfg['CENTER_COR'], roi_radius=50)
            fp, fn, det_freq, lat_error, lon_error, id_switch, id_consistency, mota = evaluator.evaluate(
                visualize=args.visualize)
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
    print('Overall performance on vehicle: ')
    print('-------------------------------------------------------')
    print('---------- Overall Evaluation Results -----------------')
    print('-------------------------------------------------------')
    print(f'False Positive rate\t\t: {mean_fp:.4f}')
    print(f'False Negative rate\t\t: {mean_fn:.4f}')
    print(f'Actual detection frequency\t: {mean_det_freq:.4f}')
    print(f'Lateral error\t\t\t: {mean_lat_error:.4f}')
    print(f'Longitudinal error\t\t: {mean_lon_error:.4f}')
    print(f'System Latency\t\t\t: {args.latency:.4f}')
    print(f'ID switch\t\t\t: {mean_id_switch:.4f}')
    print(f'ID consistency\t\t\t: {mean_id_consistency:.4f}')
    print(f'MOTA\t\t\t\t: {mean_mota:.4f}')
    print('-------------------------------------------------------')
    print('---------- End of Evaluation Results ------------------')
    print('-------------------------------------------------------')


def ped_evaluation(data_dir, args, cfg):
    global PED_LATENCY
    PED_LATENCY = args.latency
    if PED_LATENCY is None:
        print('Estimating system pedestrian detection latency...')
        # TODO: ADD EXPERIMENT TO PED DETECTION LATENCY
    print(f'Estimated system pedestrian detection latency: {PED_LATENCY} s')
    # evaluate perception system on single-vehicle trips
    print('Evaluating system perception performance on ped...')

    m, d, y = data_dir.split('/')[-1].split('-')
    local_date_str = '-'.join([y, m, d])
    fp_list = []
    fn_list = []
    det_freq_list = []
    lat_error_list = []
    lon_error_list = []
    id_switch_list = []
    id_consistency_list = []
    mota_list = []
    for trial_id in cfg['PED_TRIAL_IDS']:
        ped_det_file_path = get_detection_file_path(
            args.system, data_dir, trial_id)
        ped_rtk_file_path = f'{data_dir}/gts/ped_{trial_id}.txt'
        ped_rtk_data = PedRTKData(
            ped_rtk_file_path, plot_name='PED trajectory by RTK', date_str=local_date_str)
        detection_data = DetectionData(ped_det_file_path,
                                       plot_name='PED trajectory by detection system',
                                       detection_type=args.system)

        print(
            f'++++++++++++++++++ For trip {trial_id}: Pedestrian +++++++++++++++++++')
        # evaluate accuracy
        evaluator = Evaluator(args.system, 'ped', 1.5, detection_data, ped_rtk_data, latency=PED_LATENCY,
                              det_freq=cfg['DET_FREQUENCY'], gt_freq=cfg['GROUND_TRUTH_FREQUENCY_PED'], center_latlng=cfg['CENTER_COR'],
                              roi_radius=50)
        fp, fn, det_freq, lat_error, lon_error, id_switch, id_consistency, mota = evaluator.evaluate(
            visualize=args.visualize)
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
    print('Overall performance on pedestrian: ')
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
    args = parser.parse_args()
    testing_root_dir = os.path.join(args.data_root, args.data_name)
    testing_config_path = os.path.join(testing_root_dir, 'config.yaml')
    cfg = yaml.safe_load(open(testing_config_path))
    veh_evaluation(testing_root_dir, args, cfg)
    if cfg['PED_TRIAL_IDS'] != []:
        ped_evaluation(testing_root_dir, args, cfg)
