import re
import pandas as pd
from pprint import pprint
from utils.load_results import load_data

ascFile = r'C:\Users\Chris\Documents\GitHub\nfb\pynfb\results\0-nfb_task_test4_09-23_13-37-09\eye_nfb_2022_09_23_13_37\eye_data.asc'
h5file = r'C:\Users\Chris\Documents\GitHub\nfb\pynfb\results\0-nfb_task_test3_09-23_12-13-49\experiment_data.h5'

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
component_list = ['tacs',
                  'fcb',
                  'nfb']
# Strips the newline character
for line in Lines:
    count += 1
    if 'PROTOCOL' in line:
        print("Line{}: {}".format(count, line.strip()))
        if 'START' in line:
            for cmp in component_list:
                if cmp in line:
                    protocol_id = line.split()[2].split('_')[1].split('-')[0]
                    print(protocol_id)
                    start = int(line.split()[1])
                    asc_data[protocol_id] = {cmp: {'START':start}}
        if 'END' in line:
            for cmp in component_list:
                if cmp in line:
                    protocol_id = line.split()[2].split('_')[1].split('-')[0]
                    end = int(line.split()[1])
                    asc_data[protocol_id][cmp]['END'] = end
                    # if f'{cmp}_END' in asc_data[protocol_id]:
                    #     asc_data[protocol_id][f'{cmp}_LENGTH'] = asc_data[protocol_id][f'{cmp}_END'] - asc_data[protocol_id][f'{cmp}_START']
pprint(asc_data)

prev_end = 0
for protocol in list(asc_data.keys()):
    if 'END' in list(asc_data[protocol].values())[0]:
        length = list(asc_data[protocol].values())[0]['END'] - prev_end
        print(length)
        prev_end = list(asc_data[protocol].values())[0]['END']

# check with the h5 file
df1, fs, channels, p_names = load_data(h5file)
df1['sample'] = df1.index