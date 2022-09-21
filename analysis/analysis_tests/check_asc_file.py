import re
import pandas as pd
from pprint import pprint

ascFile = r"C:\Users\Chris\Documents\GitHub\nfb\psychopy_tasks\eye_track_data\99_psnr_2022_09_19_19_18.asc"
ascFile = r'C:\Users\Chris\Documents\GitHub\nfb\psychopy_tasks\eye_track_data\99_psnr_2022_09_21_17_23.asc'

file1 = open(ascFile, 'r')
Lines = file1.readlines()

trial_start = 0
trial_end = 0
trial_id = 0
fc_start = 0

first = False
offset = 0
count = 0
asc_data = {}
component_list = ['fc',
                  'left_probe',
                  'right_probe',
                  'left_cue',
                  'right_cue',
                  'centre_cue1',
                  'centre_cue2',
                  'stim',
                  'key_resp',
                  'key_log']
# Strips the newline character
for line in Lines:
    count += 1
    if 'TRIAL' in line:
        if 'trials' in line and 'START' in line:
            trial_start = int(line.split(' ')[0].split('\t')[1])
            trial_id = line.split(' ')[1].split('_')[1]
            asc_data[trial_id] = {}
            # print("Line{}: {}".format(count, line.strip()))
        if 'START' in line:
            for cmp in component_list:
                if cmp in line:
                    asc_data[trial_id][cmp] = {}
                    start = int(line.split(' ')[0].split('\t')[1])
                    asc_data[trial_id][cmp]['start'] = start
        if 'END' in line:
            for cmp in component_list:
                if cmp in line:
                    end = int(line.split(' ')[0].split('\t')[1])
                    asc_data[trial_id][cmp]['end'] = end
                    asc_data[trial_id][cmp]['length'] = asc_data[trial_id][cmp]['end'] - asc_data[trial_id][cmp]['start']
pprint(asc_data)

key = 'start'
for k, v in asc_data.items():
    for kk, vv in v.items():
        print(f"{k}: {kk}: {key}: {vv[key]}")