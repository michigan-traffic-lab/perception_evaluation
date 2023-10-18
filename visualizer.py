import plotly.graph_objects as go
# import matplotlib
# import matplotlib.cm as cmx
# import numpy as np
import json
import argparse

cmaps = ['cyan', 'green', 'yellow', 'orange', 'pink', 'purple']

class Plotter:
    def __init__(self, center_lat, center_lon, title='CAV Trajectories Recorded by RTK and Detected by CI'):
        self.fig = go.Figure()     
        self.center_lat = center_lat 
        self.center_lon = center_lon
        self.fig.update_layout(
            mapbox={
                'accesstoken': 'pk.eyJ1Ijoiemh1aGoiLCJhIjoiY2ttdGF2bDd1MHEwbjJybzVxMTduOHVjbSJ9.wT_hzXAry0m33OrdaeTzRA',
                'style': "satellite-streets", 'center': {'lat': (self.center_lat), 'lon': (self.center_lon)},
                'zoom': 20},
            showlegend=True, title_text=title)
        return
        
    def plot_traj_data(self, dp_list, plot_name, color=None):
        # need to support multiple vehicles
        ids = [dtdp.id for dtdp in dp_list]
        ids_set = set(ids)
        for idx, id in enumerate(ids_set):
            dp_sublist = [dp for dp in dp_list if dp.id == id]
            self.fig.add_trace(go.Scattermapbox(
                lat=[dp.lat for dp in dp_sublist], lon=[dp.lon for dp in dp_sublist], mode='markers', text=[str(dp) for dp in dp_sublist],
                marker={'size': 10, 'color': color if color is not None else cmaps[hash(str(id)) % len(cmaps)]}, name=plot_name+f'ID: {id}'))

    def plot_matching(self, dtdp_list, color='yellow'):
        for dtdp_idx, dtdp in enumerate(dtdp_list):
            # if match_indices is None and dtdp.match != -1:
            #     matched_gtdp = gtdp_list[dtdp.match]
            # elif match_indices is None and dtdp.match == -1:
            #     print('A valid match idx is required')
            #     raise ValueError
            # else:
            #     matched_gtdp = gtdp_list[match_indices[dtdp_idx]]

            if dtdp.point_wise_match is None:
                continue
            matched_gtdp = dtdp.point_wise_match
            self.fig.add_trace(go.Scattermapbox(lat=[matched_gtdp.lat, dtdp.lat], lon=[matched_gtdp.lon, dtdp.lon], mode='lines',
            marker={'size': 5, 'color': color}, name='matching'))

if __name__ == '__main__':
    from utils import DataPoint
    argparser = argparse.ArgumentParser(description='Visualize the trajectory data.')
    argparser.add_argument('-i', '--input', type=str, help='the input JSON file.')
    args = argparser.parse_args()
    center_lat = 42.30092239379880
    center_lon = -83.69866180419920
    with open(args.input, 'r') as f:
        data = json.load(f)
    dtdp_list = []
    for d in data:
        for obj in d['objs']:
            dtdp_list.append(DataPoint(id=obj['id'], lat=obj['lat'], lon=obj['lon'], time=d['timestamp'] * 1e9))
    # print(len(dtdp_list))
    plotter = Plotter(center_lat, center_lon)
    plotter.plot_traj_data(dtdp_list, 'custom')
    plotter.fig.show()
