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
        
    def plot_traj_data(self, dp_list, plot_name, color='blue'):
        # need to support multiple vehicles
        self.fig.add_trace(go.Scattermapbox(
            lat=[dp.lat for dp in dp_list], lon=[dp.lon for dp in dp_list], mode='markers', text=[str(dp) for dp in dp_list],
            marker={'size': 5, 'color': color}, name=plot_name))

    def plot_matching(self, dtdp_list, gtdp_list, color='yellow'):
        for dtdp in dtdp_list:
            if dtdp.match != -1:
                matched_gtdp = gtdp_list[dtdp.match]
                self.fig.add_trace(go.Scattermapbox(lat=[matched_gtdp.lat, dtdp.lat], lon=[matched_gtdp.lon, dtdp.lon], mode='lines',
                marker={'size': 10, 'color': color}, name='matching'))