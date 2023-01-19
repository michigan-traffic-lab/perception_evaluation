import plotly.graph_objects as go
import numpy as np

class Plotter:
    def __init__(self, center_lat, center_lon):
        self.fig = go.Figure()     
        self.center_lat = center_lat 
        self.center_lon = center_lon
        self.fig.update_layout(
            mapbox={
                'accesstoken': 'pk.eyJ1Ijoiemh1aGoiLCJhIjoiY2ttdGF2bDd1MHEwbjJybzVxMTduOHVjbSJ9.wT_hzXAry0m33OrdaeTzRA',
                'style': "satellite-streets", 'center': {'lat': (self.center_lat), 'lon': (self.center_lon)},
                'zoom': 20},
            showlegend=True, title_text='CAV Trajectories Recorded by RTK and Detected by CI')
        return   
        
    def plot_traj_data(self, dp_list, plot_name, color=None):
        # need to support multiple vehicles
        ids = [dtdp.id for dtdp in dp_list]
        ids_set = set(ids)
        for idx, id in enumerate(ids_set):
            dp_sublist = [dp for dp in dp_list if dp.id == id]
            self.fig.add_trace(go.Scattermapbox(
                lat=[dp.lat for dp in dp_sublist], lon=[dp.lon for dp in dp_sublist], mode='markers', text=[str(dp) for dp in dp_sublist],
                marker={'size': 5, 'color': color if color is not None else idx}, name=plot_name+f'ID: {id}'))

    def plot_matching(self, dtdp_list, gtdp_list, match_indices=None, color='yellow'):
        for dtdp_idx, dtdp in enumerate(dtdp_list):
            if match_indices is None and dtdp.match != -1:
                matched_gtdp = gtdp_list[dtdp.match]
            elif match_indices is None and dtdp.match == -1:
                print('A valid match idx is required')
                raise ValueError
            else:
                matched_gtdp = gtdp_list[match_indices[dtdp_idx]]
            self.fig.add_trace(go.Scattermapbox(lat=[matched_gtdp.lat, dtdp.lat], lon=[matched_gtdp.lon, dtdp.lon], mode='lines',
            marker={'size': 10, 'color': color}, name='matching'))