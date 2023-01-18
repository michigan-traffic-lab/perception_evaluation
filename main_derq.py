import argparse
from utils_mcity_legacy import CAVData, DetectionData
from evaluator import Evaluator

GROUND_TRUTH_FREQUENCY=50 #50hz
DET_FREQUENCY=2.5 #10hz Derq system  Conti
DISTANCE_THRESHOLD=1.5 #1.5 meters
LATENCY=0.1984 #change to real value when calculate longitudinal errors
CENTER_COR=[42.300947, -83.698649]

parser = argparse.ArgumentParser()
parser.add_argument('--trial-name', default='8')
parser.add_argument('--experiment-name', default='11-15-2022')
parser.add_argument('--config-file', '-c', help="configuration file path", default='experiment_settings.yml')
args = parser.parse_args()

# det_traj_file = './pred_ne_v2_filtered_8.csv'
# det_traj_file = './data/pred_filtered_4gs_v2_1.csv'
det_traj_file = 'data/mcity-perception-data/TestingResult/12-19-2022/1/edge_DSRC_BSM_send_1.csv'
cav_1_rtk_file = 'data/mcity-perception-data/TestingResult/12-19-2022/1/gps_fix_1.txt'
cav_1_speed_file = 'data/mcity-perception-data/TestingResult/12-19-2022/1/gps_vel_1.txt'


if __name__  == '__main__':
    vehicle_data = [
        CAVData([cav_1_rtk_file, cav_1_speed_file], plot_name='CAV 1 trajectory recorded by RTK', gps_type='oxford'),
    ]
    detection_data = [
        DetectionData(det_traj_file, plot_name='detection system 1', detection_type='Derq'),
    ]

    ## evaluate accuracy
    evaluator = Evaluator(1.5, detection_data[0], vehicle_data[0], latency=LATENCY, 
                          det_freq=DET_FREQUENCY, gt_freq=GROUND_TRUTH_FREQUENCY, center_lat=CENTER_COR[0], center_lon=CENTER_COR[1])
    evaluator.evaluate(visualize=True)
    