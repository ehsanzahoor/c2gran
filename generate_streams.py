'''
This script reads the server csv file for start/stop time of each stream and creates the stream according to the start times
The stream ends with the specified end time. 
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
