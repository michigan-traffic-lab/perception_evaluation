# -*- coding: utf-8 -*-

from datetime import date, datetime, timedelta
from dateutil import tz

DATE_FORMAT = '%Y-%m-%d'
TIME_FORMAT = '%H:%M:%S.%f'
DATE_TIME_FORMAT = f'{DATE_FORMAT} {TIME_FORMAT}'

EPOCH = datetime.strptime('1980-01-06', DATE_FORMAT)

def add_leading_zero(x):
    if float(x) < 100000:
        str_x = str(x)
        str_x.zfill(6)
    return str_x

def utc_date_str_to_gps_weeks(utc_date_str):
    dt = datetime.strptime(utc_date_str, DATE_FORMAT)
    gps_weeks = (dt - EPOCH).days // 7
    return gps_weeks


def utc_date_str_to_leap_seconds(utc_date_str):
    if utc_date_str <= '2016-12-31':
        raise ValueError('Please refer '
                         'https://en.racelogic.support/LabSat_GNSS_Simulators/Knowledge_Base/General_Info/LabSat_Leap_Second_Guide')
    return 18


def gps_weeks_seconds_to_utc_date_time_str(gps_weeks: int, gps_sow: float, leap_sec: int):
    dt = EPOCH + timedelta(weeks=gps_weeks) + timedelta(seconds=gps_sow) - timedelta(seconds=leap_sec)
    return dt.strftime(DATE_TIME_FORMAT)


def utc_date_time_str_to_local_date_time_str(utc_date_time_str):
    from_zone = tz.gettz('UTC')
    to_zone = tz.gettz('America/Detroit')
    utc = datetime.strptime(utc_date_time_str, DATE_TIME_FORMAT)
    utc = utc.replace(tzinfo=from_zone)
    return utc.astimezone(to_zone).strftime(DATE_TIME_FORMAT)


def utc_date_time_str_to_timestamp(utc_date_time_str):
    return datetime.strptime(utc_date_time_str, DATE_TIME_FORMAT).replace(tzinfo=tz.gettz('UTC')).timestamp()


def utc_date_str_gps_sow_to_local_date_time_str(utc_date_str, gps_sow):
    gps_weeks = utc_date_str_to_gps_weeks(utc_date_str)
    leap_sec = utc_date_str_to_leap_seconds(utc_date_str)
    utc_date_time_str = gps_weeks_seconds_to_utc_date_time_str(gps_weeks, gps_sow, leap_sec)
    print(utc_date_time_str)
    return utc_date_time_str_to_local_date_time_str(utc_date_time_str)


def local_date_str_utc_hms_float_to_local_date_time_str(local_date_str, utc_hms_float):
    utc_hms_str = str(utc_hms_float)
    utc_date_time_str = f'{local_date_str} {utc_hms_str[0:2]}:{utc_hms_str[2:4]}:{utc_hms_str[4:]}'
    return utc_date_time_str_to_local_date_time_str(utc_date_time_str)


def local_date_str_utc_hms_float_to_timestamp(local_date_str, utc_hms_float):
    utc_hms_float = float(utc_hms_float)
    # utc_hms_float
    # utc_hms_str = str(add_leading_zero(utc_hms_float))
    # print(utc_hms_str)
    hours = int(utc_hms_float//10000)
    hours = str(hours).zfill(2)
    minutes = int((utc_hms_float%10000) // 100)
    minutes = str(minutes).zfill(2)
    seconds = round((utc_hms_float%100),1)
    # print(str(seconds))
    x = str(seconds).split(".")
    # print(seconds)
    seconds = x[0].zfill(2)+"."+x[1]
    # print(seconds)
    utc_date_time_str = f'{local_date_str} {hours}:{minutes}:{seconds}'
    # print(utc_date_time_str)
    return utc_date_time_str_to_timestamp(utc_date_time_str)


def utc_hms_float_to_seconds_in_minutes(utc_hms_float):
    utc_hms_str = str(utc_hms_float)
    return float(utc_hms_str[4:])


if __name__ == '__main__':
    print(local_date_str_utc_hms_float_to_local_date_time_str('2022-12-19', 154311.50))
    print(utc_hms_float_to_seconds_in_minutes(154311.50))
    print(local_date_str_utc_hms_float_to_timestamp('2023-01-05', 190840.00))
