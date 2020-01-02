'''
This script reads the server csv file for start/stop time of each stream and creates the stream according to the start times
The stream ends with the specified end time. 
The CSV file is in this format:
call_start_datetime,call_end_datetime,cpu_load_weight,memory_change_percentage,bytes_transmitted_mb,duration_seconds,hour,Day_of_the_Week,weekday_name
2019-10-08 00:00:06,2019-10-08 00:44:05,1,3.2,33.0,2639.0,0,1,Tuesday
2019-10-08 00:00:14,2019-10-08 00:01:33,1,0.2,4.0,79.0,0,1,Tuesday
2019-10-08 00:00:18,2019-10-08 00:23:45,1,5.6,25.0,1407.0,0,1,Tuesday
2019-10-08 00:00:19,2019-10-08 00:09:54,1,1.8,10.0,575.0,0,1,Tuesday
2019-10-08 00:00:28,2019-10-08 00:02:47,1,0.4,6.0,139.0,0,1,Tuesday
........
........
'''
import time
import datetime
from threading import Thread
import pandas as pd
import numpy as np
import subprocess


def start_stream(duration):
    command = "vlc http://10.154.50.83:8080"
    print("Thread started for duration %d"%duration)
    #stat = subprocess.check_output(command.split())
    p = subprocess.Popen(command.split())
    time.sleep(duration)
    p.terminate()
    print("closing")

df = pd.read_csv("server_1.csv")
# Change the data types of some columns from object to datime
df['call_end_datetime'] = pd.to_datetime(df['call_end_datetime'])
df['call_start_datetime'] = pd.to_datetime(df['call_start_datetime'])

#t = Thread(target=start_stream, args=[10])
#t.start()

prev_time = df['call_start_datetime'][0]
all_threads = []
for ind in df.index:
    curr_time = df['call_start_datetime'][ind]
    time.sleep((curr_time-prev_time) / np.timedelta64(1, 's'))  # wait before initiating next thread
    #print((curr_time-prev_time) / np.timedelta64(1, 's'), df['duration_seconds'][ind])
    t = Thread(target=start_stream, args=[df['duration_seconds'][ind]])
    all_threads.append(t)
    t.start()
    prev_time = df['call_start_datetime'][ind]


print("closed")
