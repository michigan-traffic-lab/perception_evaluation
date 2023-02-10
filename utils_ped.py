# -*- coding: utf-8 -*-
import pandas as pd

from time_converter import local_date_str_utc_hms_float_to_local_date_time_str, \
    local_date_str_utc_hms_float_to_timestamp


class PedData:
    def __init__(self):
        self.longs = []
        self.lats = []
        self.datetime = []
        self.texts = []
        self.speeds = []
        self.exist = True
        self.df = None
        self.detection_type = 'ped'


class PedRTKData(PedData):
    def __init__(self, file_path, plot_name, date_str):
        super().__init__()
        self.file_path = file_path
        self.plot_name = plot_name
        self.date_str = date_str
        self.load_data_from_csv()

    def load_data_from_csv(self):
        df = pd.read_csv(self.file_path, header=None)
        # print(df)
        df.rename(columns={
            0: 'hhmmss',
            1: 'lat',
            2: 'lng',
            3: 'type',
        }, inplace=True)
        df['sec'] = df['hhmmss'].apply(lambda x: local_date_str_utc_hms_float_to_timestamp(self.date_str, x)*1e9)
        self.datetime = df['sec'].tolist()
        self.longs = df['lng'].tolist()
        self.lats = df['lat'].tolist()
        self.texts = [f'Date time str from RTK: {_}' for _ in self.datetime]
        self.df = df

    def update_texts(self):
        self.texts = [f'Date time str from RTK: {_}' for _ in self.datetime]



