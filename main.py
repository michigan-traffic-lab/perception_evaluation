import argparse
from utils_mcity_legacy import CAVData, DetectionData
from evaluator import Evaluator
import numpy as np
from utils import get_detection_file_path, prepare_data, prepare_gt_data
import os
import yaml
from pathlib import Path
from visualizer import Plotter

from utils_ped import PedRTKData

import sys

assert sys.version_info >= (
    3, 7), 'Please use Python 3.7 or higher, Python version before 3.7 will NOT produce the correct results.'

parser = argparse.ArgumentParser(prog='EvaluationSettingParser',
                                 description='This parser specifies the settings for \
                                              the perception evaluation program')
parser.add_argument('-d', '--data-root',
                    type=str,
                    # required=True,
                    default='./data',
                    help='the root dir of the testing data')
parser.add_argument('-n', '--data-name',
                    required=True,
                    type=str,
                    help='the name of the testing, often represented by its testing date')
parser.add_argument('-s', '--system',
                    type=str,
                    # required=True,
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

evaluator = None
cfg = None


def veh_evaluation(data_dir, args, cfg):
    global evaluator
    # evaluate system latency
    if args.latency == -1:
        if cfg['LATENCY_TRIAL_IDS'] == []:
            raise ValueError('A valid LATENCY_TRIAL_IDS is required')
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
            raw_dtdps, raw_gtdps = prepare_data(
                detection_data, vehicle_data, cate=evaluator.cate, source=evaluator.source)
            evaluator = Evaluator(args.system, 'veh', 1.5, latency=None, det_freq=cfg['DET_FREQUENCY'],
                                  gt_freq=cfg['GROUND_TRUTH_FREQUENCY_VEH1'], center_latlng=cfg['CENTER_COR'], roi_radius=50)
            dtdps, gtdps = evaluator.remove_outside_data(
                raw_dtdps, raw_gtdps, radius=20)
            # gtdps.trajectories[0].sample_rate = cfg['GROUND_TRUTH_FREQUENCY_VEH1']

            print(
                f'++++++++++++++++++ For trip {trial_id} +++++++++++++++++++')
            # TODO: change vehicle_data to gt_data

            latency = evaluator.compute_latency(dtdps, gtdps)
            print(f'Estimated system vehicle detection latency: {latency} s')
            latency_list.append(latency)
        # args.latency = np.mean(np.array(latency_list))
        latency = np.mean(np.array(latency_list))
        evaluator.latency = latency
        print(f'Estimated system vehicle detection latency: {latency} s')
    else:
        evaluator.latency = args.latency

    # evaluate perception system on single-vehicle trips
    print('Evaluating system perception performance on single-vehicle scenerios...')

    if cfg['SINGLE_VEH_TRIAL_IDS'] != []:
        for trial_id in cfg['SINGLE_VEH_TRIAL_IDS']:
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
            raw_dtdps, raw_gtdps = prepare_data(
                detection_data, vehicle_data, cate=evaluator.cate, source=evaluator.source)
            dtdps, gtdps = evaluator.remove_outside_data(
                raw_dtdps, raw_gtdps, inplace=False)
            evaluator.clear_match(dtdps)
            evaluator.clear_match(gtdps)

            gtdps.trajectories[0].sample_rate = float(
                cfg['GROUND_TRUTH_FREQUENCY_VEH1'])

            tp, fp, fn, fp_rate, fn_rate = evaluator.match_points(
                dtdps, gtdps)
            print('true positive: ', tp, 'false positive: ',
                  fp, 'false negative: ', fn)
            print('false positive rate: ', fp_rate,
                  'false negative rate: ', fn_rate)

            ids = evaluator.compute_id_switch(dtdps, gtdps)
            print(f"number of ID switch: {ids}")

            mota = evaluator.compute_mota(
                fp, fn, ids, len(dtdps.dp_list))
            print(f"MOTA: {mota}")

            motp = evaluator.compute_motp(dtdps)
            print(f"MOTP: {motp}")

            evaluator.clear_match(dtdps)
            evaluator.clear_match(gtdps)

            gtdps.trajectories[0].sample_rate = float(
                cfg['GROUND_TRUTH_FREQUENCY_VEH1'])
            match_result, tpa, fpa, fna = evaluator.match_trajectories(
                dtdps, gtdps)

            idf1 = evaluator.compute_idf1(tpa, fpa, fna)
            print(f"IDF1: {idf1}")
            hota = evaluator.compute_hota(tpa, fpa, fna, tp, fp, fn)
            print(f"HOTA: {hota}")

            # # plotter.plot_matching(fn_dtdp_list, fn_gtdp_list, color='green')
            if args.visualize:
                plotter = Plotter(center_lat=evaluator.center_lat,
                                  center_lon=evaluator.center_lon,
                                  title=f'Single Vehicle detection results for trip {trial_id}')
                plotter.plot_traj_data(
                    gtdps.dp_list, plot_name='GPS RTK', color='blue')
                plotter.plot_traj_data(
                    dtdps.dp_list, plot_name='Msight detection', color='red')
                plotter.fig.show()


    # evaluate perception system on two-vehicle trips
    if cfg['TWO_VEH_TRIAL_IDS'] != []:
        print('Evaluating system perception performance on two-vehicle scenerios...')
        for trial_id in cfg['TWO_VEH_TRIAL_IDS']:
            print(
                f'+++++++++++++++++++++++++++ For Trip {trial_id} ++++++++++++++++++++++++++++')
            det_traj_file = get_detection_file_path(
                args.system, data_dir, trial_id)
            cav_1_rtk_file = f'{data_dir}/gts/gps_fix_veh1_{trial_id}.txt'
            cav_1_speed_file = f'{data_dir}/gts/gps_vel_veh1_{trial_id}.txt'

            cav_2_rtk_file = f'{data_dir}/gts/gps_fix_veh2_{trial_id}.txt'
            cav_2_speed_file = f'{data_dir}/gts/gps_vel_veh2_{trial_id}.txt'

            vehicle1_data = CAVData([cav_1_rtk_file, cav_1_speed_file], plot_name='CAV 1 trajectory recorded by RTK',
                                    gps_type='oxford')
            vehicle2_data = CAVData([cav_2_rtk_file, cav_2_speed_file], plot_name='CAV 2 trajectory recorded by RTK',
                                    gps_type='oxford')
            detection_data = DetectionData(
                det_traj_file, plot_name='detection system 1', detection_type=args.system)
            raw_dtdps, _ = prepare_data(
                detection_data, vehicle1_data, cate=evaluator.cate, source=evaluator.source)
            raw_gtdps = prepare_gt_data(vehicle1_data, vehicle2_data, cfg)
            # print(len(raw_gtdps.dataframes))

            dtdps, gtdps = evaluator.remove_outside_data(
                raw_dtdps, raw_gtdps, inplace=False)

            evaluator.clear_match(dtdps)
            evaluator.clear_match(gtdps)

            gtdps.trajectories[0].sample_rate = float(
                cfg['GROUND_TRUTH_FREQUENCY_VEH1'])
            gtdps.trajectories[1].sample_rate = float(
                cfg['GROUND_TRUTH_FREQUENCY_VEH1'])

            tp, fp, fn, fp_rate, fn_rate = evaluator.match_points(
                dtdps, gtdps)
            print('true positive: ', tp, 'false positive: ',
                  fp, 'false negative: ', fn)
            print('false positive rate: ', fp_rate,
                  'false negative rate: ', fn_rate)

            ids = evaluator.compute_id_switch(dtdps, gtdps)
            print(f"number of ID switch: {ids}")

            mota = evaluator.compute_mota(
                fp, fn, ids, len(dtdps.dp_list))
            print(f"MOTA: {mota}")

            motp = evaluator.compute_motp(dtdps)
            print(f"MOTP: {motp}")

            match_result, tpa, fpa, fna = evaluator.match_trajectories(
                dtdps, gtdps)

            # print(match_result, tpa)
            idf1 = evaluator.compute_idf1(tpa, fpa, fna)
            print(f"IDF1: {idf1}")
            hota = evaluator.compute_hota(tpa, fpa, fna, tp, fp, fn)
            print(f"HOTA: {hota}")

            # # plotter.plot_matching(fn_dtdp_list, fn_gtdp_list, color='green')
            if args.visualize:
                plotter = Plotter(center_lat=evaluator.center_lat,
                                  center_lon=evaluator.center_lon,
                                  title=f'Two Vehicle detection results for trip {trial_id}')
                plotter.plot_traj_data(
                    gtdps.dp_list, plot_name='GPS RTK', color='blue')
                plotter.plot_traj_data(
                    dtdps.dp_list, plot_name='Msight detection', color='red')
                plotter.fig.show()


def ped_evaluation(data_dir, args, cfg):
    if cfg['PED_TRIAL_IDS'] == []:
        return
    evaluator.cate = 'ped'
    evaluator.roi_radius = 30
    # global PED_LATENCY
    # PED_LATENCY = args.latency
    # if PED_LATENCY is None:
    #     print('Estimating system pedestrian detection latency...')
    # TODO: ADD EXPERIMENT TO PED DETECTION LATENCY
    # print(f'Estimated system pedestrian detection latency: {PED_LATENCY} s')
    # evaluate perception system on single-vehicle trips
    print('Evaluating system perception performance on ped...')

    local_date_str = cfg['DATE']
    for trial_id in cfg['PED_TRIAL_IDS']:
        ped_det_file_path = get_detection_file_path(
            args.system, data_dir, trial_id, object="ped")
        ped_rtk_file_path = f'{data_dir}/gts/ped_{trial_id}.txt'
        ped_rtk_data = PedRTKData(
            ped_rtk_file_path, plot_name='PED trajectory by RTK', date_str=local_date_str)
        detection_data = DetectionData(ped_det_file_path,
                                       plot_name='PED trajectory by detection system',
                                       detection_type=args.system)
        raw_dtdps, raw_gtdps = prepare_data(
            detection_data, ped_rtk_data, cate=evaluator.cate, source=evaluator.source)
        dtdps, gtdps = evaluator.remove_outside_data(
            raw_dtdps, raw_gtdps, inplace=False)
        # print('number of pedestrian:', gtdps.num_traj, 'first pedestrian name:', list(gtdps.trajectories.keys())[0])
        gtdps.trajectories[0].sample_rate = float(
            cfg['GROUND_TRUTH_FREQUENCY_PED'])

        print(
            f'++++++++++++++++++ For trip {trial_id}: Pedestrian +++++++++++++++++++')
        evaluator.clear_match(dtdps)
        evaluator.clear_match(gtdps)
        tp, fp, fn, fp_rate, fn_rate = evaluator.match_points(
            dtdps, gtdps)
        print('true positive: ', tp, 'false positive: ',
                fp, 'false negative: ', fn)
        print('false positive rate: ', fp_rate,
                'false negative rate: ', fn_rate)

        ids = evaluator.compute_id_switch(dtdps, gtdps)
        print(f"number of ID switch: {ids}")

        mota = evaluator.compute_mota(
            fp, fn, ids, len(dtdps.dp_list))
        print(f"MOTA: {mota}")

        motp = evaluator.compute_motp(dtdps)
        print(f"MOTP: {motp}")

        match_result, tpa, fpa, fna = evaluator.match_trajectories(
            dtdps, gtdps)

        idf1 = evaluator.compute_idf1(tpa, fpa, fna)
        print(f"IDF1: {idf1}")
        hota = evaluator.compute_hota(tpa, fpa, fna, tp, fp, fn)
        print(f"HOTA: {hota}")


        if args.visualize:
            plotter = Plotter(center_lat=evaluator.center_lat,
                              center_lon=evaluator.center_lon,
                              title=f'Pedestrian detection results for trip {trial_id}')
            plotter.plot_traj_data(
                gtdps.dp_list, plot_name='GPS RTK', color='blue')
            plotter.plot_traj_data(
                dtdps.dp_list, plot_name='Msight detection', color='red')
            plotter.fig.show()


if __name__ == '__main__':
    args = parser.parse_args()
    testing_root_dir = Path(args.data_root) / args.data_name
    testing_config_path = Path(testing_root_dir) / 'config.yaml'
    cfg = yaml.safe_load(open(testing_config_path))
    evaluator = Evaluator(args.system, 'veh', 1.5, det_freq=cfg['DET_FREQUENCY'],
                          gt_freq=cfg['GROUND_TRUTH_FREQUENCY_VEH1'], center_latlng=cfg['CENTER_COR'], roi_radius=50, latency=args.latency)
    veh_evaluation(testing_root_dir, args, cfg)
    ped_evaluation(testing_root_dir, args, cfg)
