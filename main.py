import argparse
from pathlib import Path
import yaml
from evaluator import Evaluator
from utils import read_data, compute_speed_and_heading, Result
from visualizer import Plotter


def evaluate(dtdps, gtdps, evaluator):
    frequency = evaluator.compute_actual_frequency(dtdps)
    interval_variance = evaluator.compute_interval_variance(dtdps)
    for _, t in dtdps.trajectories.items():
        compute_speed_and_heading(t.dp_list, interval=2)
    for _, t in gtdps.trajectories.items():
        compute_speed_and_heading(t.dp_list, interval=10)
    dtdps, gtdps = evaluator.remove_outside_data(
        dtdps, gtdps, inplace=False)
    evaluator.clear_match(dtdps)
    evaluator.clear_match(gtdps)

    tp, fp, fn, fp_rate, fn_rate, total_expected_detection = evaluator.match_points(
        dtdps, gtdps)
    fn_freq, fnr_freq, total_expected_detection_freq = evaluator.compute_false_negatives_by_frequency(
        dtdps, gtdps)
    # if args.visualize:
    # plotter = Plotter(center_lat=evaluator.center_lat,
    #                     center_lon=evaluator.center_lon,
    #                     title=f'Detection results for trip {trial_id}')
    # plotter.plot_traj_data(
    #     gtdps.dp_list, plot_name='GPS RTK', color='blue')
    # plotter.plot_traj_data(
    #     dtdps.dp_list, plot_name='Msight detection', color='red')
    # plotter.plot_matching(dtdps.dp_list)
    # plotter.fig.show()
    ids = evaluator.compute_id_switch(dtdps, gtdps)
    mota = evaluator.compute_mota(
        fp, fn, ids, len(dtdps.dp_list))
    mota_freq = evaluator.compute_mota(
        fp, fn_freq, ids, len(dtdps.dp_list)
    )
    motp = evaluator.compute_motp(dtdps)

    evaluator.clear_match(dtdps)
    evaluator.clear_match(gtdps)

    match_result, tpa, fpa, fna = evaluator.match_trajectories(
        dtdps, gtdps, total_expected_detection_freq)
    # print(f"tpa: {tpa}, fpa: {fpa}, fna: {fna}")

    idf1 = evaluator.compute_idf1(tpa, fpa, fna)
    hota = evaluator.compute_hota(tpa, fpa, fna, tp, fp, fn)
    return fp_rate, fn_rate, mota, mota_freq, motp, idf1, hota, ids, frequency, interval_variance, fnr_freq, dtdps, gtdps


argparser = argparse.ArgumentParser()
argparser.add_argument('-d', '--data-root', type=Path, default=Path('./data'))
argparser.add_argument('-n', '--data-name', type=str, required=True)
argparser.add_argument('-l', '--latency', type=float)
argparser.add_argument('-v', '--visualize', action='store_true')
args = argparser.parse_args()

data_root = args.data_root
data_name = args.data_name

with open(data_root / data_name / 'config.yaml', 'r') as f:
    cfg = yaml.safe_load(f)
evaluator = Evaluator('veh', cfg['TP_DISTANCE_THRESHOLD'], det_freq=cfg['DET_FREQUENCY'],
                      gt_freq=cfg['GROUND_TRUTH_FREQUENCY_VEH'], center_latlng=cfg['CENTER_COR'], roi_radius=cfg['ROI_RADIUS'], latency=args.latency)
result = Result()
for trial in cfg['trials']:
    trial_id = trial['id']
    det_path = Path(data_root) / data_name / trial['det_file']
    gt_path = Path(data_root) / data_name / trial['gt_file']
    print(f"   ++++++++ Processing trial {trial_id} ++++++++")
    if trial['type'] == 'latency' and args.latency is not None:
        continue
    elif trial['type'] == 'latency':
        dtdps = read_data(det_path, kind='veh')
        gtdps = read_data(gt_path)
        dtdps, gtdps = evaluator.remove_outside_data(
            dtdps, gtdps, radius=trial['roi_radius'])
        latency = evaluator.compute_latency(dtdps, gtdps)
        result.add_trial({
            'id': trial_id,
            'type': 'latency',
            'latency': latency
        })
        print(f"Latency: {latency}")
    elif trial['type'] == 'vehicle_evaluation':
        if 'roi_radius' in trial:
            evaluator.roi_radius = trial['roi_radius']
        else:
            evaluator.roi_radius = cfg['ROI_RADIUS']

        if args.latency is not None:
            evaluator.latency = args.latency
        else:
            evaluator.latency = result.latency
        assert evaluator.latency is not None, "Latency is not set"
        print(f'using latency of {evaluator.latency}')

        # evaluator.det_freq = cfg['DET_FREQUENCY']
        evaluator.gt_freq = cfg['GROUND_TRUTH_FREQUENCY_VEH']

        dtdps = read_data(det_path, kind='veh')
        gtdps = read_data(gt_path, kind='veh')
        fp_rate, fn_rate, mota, mota_freq, motp, idf1, hota, ids, freq, interval_variance, fnr_freq, _, _ = evaluate(
            dtdps, gtdps, evaluator)
        result.add_trial({
            'id': trial_id,
            'type': 'vehicle_evaluation',
            'fp_rate': fp_rate,
            'fn_rate': fn_rate,
            'mota': mota,
            'mota_freq': mota_freq,
            'motp': motp,
            'idf1': idf1,
            'hota': hota,
            'ids': ids,
            'frequency': freq,
            'interval_variance': interval_variance,
            'fnr_freq': fnr_freq
        })
        print(
            f"False positive rate: {fp_rate}, False negative rate: {fn_rate}, MOTA: {mota}, MOTP: {motp}, IDF1: {idf1}, HOTA: {hota}, ID Switch: {ids}")
    elif trial['type'] == 'pedestrian_evaluation':
        if args.latency is not None:
            evaluator.latency = args.latency
        else:
            evaluator.latency = result.latency
        assert evaluator.latency is not None, "Latency is not set"
        print(f'using latency of {evaluator.latency}')
        dtdps = read_data(det_path, kind='ped')
        gtdps = read_data(gt_path, kind='ped')
        if 'roi_radius' in trial:
            evaluator.roi_radius = trial['roi_radius']
        else:
            evaluator.roi_radius = cfg['ROI_RADIUS']
        # evaluator.det_freq = cfg['DET_FREQUENCY']
        evaluator.gt_freq = cfg['GROUND_TRUTH_FREQUENCY_PED']

        fp_rate, fn_rate, mota, mota_freq, motp, idf1, hota, ids, freq, interval_variance, fnr_freq, dtdps, gtdps = evaluate(
            dtdps, gtdps, evaluator)
        result.add_trial({
            'id': trial_id,
            'type': 'pedestrian_evaluation',
            'fp_rate': fp_rate,
            'fn_rate': fn_rate,
            'mota': mota,
            'mota_freq': mota_freq,
            'motp': motp,
            'idf1': idf1,
            'hota': hota,
            'ids': ids,
            'frequency': freq,
            'interval_variance': interval_variance,
            'fnr_freq': fnr_freq
        })
        print(
            f"False positive rate: {fp_rate}, False negative rate: {fn_rate}, MOTA: {mota}, MOTP: {motp}, IDF1: {idf1}, HOTA: {hota}, ID Switch: {ids}")
        ####### visualization########
    if args.visualize:
        plotter = Plotter(center_lat=evaluator.center_lat,
                          center_lon=evaluator.center_lon,
                          title=f'Detection results for trip {trial_id}')
        plotter.plot_traj_data(
            gtdps.dp_list, plot_name='GPS RTK', color='blue')
        plotter.plot_traj_data(
            dtdps.dp_list, plot_name='Msight detection', color='red')
        # plotter.plot_matching(dtdps.dp_list)
        plotter.fig.show()
print(result)
