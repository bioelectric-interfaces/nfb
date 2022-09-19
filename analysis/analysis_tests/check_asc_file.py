import re

ascFile = r"C:\Users\Chris\Documents\GitHub\nfb\psychopy_tasks\eye_track_data\99_psnr_2022_09_19_19_18.asc"

file1 = open(ascFile, 'r')
Lines = file1.readlines()

trial_start = 0
trial_end = 0
trial_id = 0
fc_start = 0

first = False
offset = 0

# def parse_component(line, component_name, offset):
#     start = 0
#     end = 0
#     id = 0
#     if 'fc' in line and 'START' in line:
#         start = int(line.split(' ')[0].split('\t')[1])
#     elif 'fc' in line and 'END' in line:
#         end = int(line.split(' ')[0].split('\t')[1])
#         length = int(end) - int(start)
#         print(f'{id} fc start: {start - offset}, end: {end - offset}, length: {length}')

count = 0
# Strips the newline character
for line in Lines:
    count += 1
    if not first and 'MSG' in line:
        print("Line{}: {}".format(count, line.strip()))
        offset = int(line.split()[1])
        first = True
    if 'TRIAL' in line:
        # print("Line{}: {}".format(count, line.strip()))
        if 'trials' in line and 'START' in line:
            trial_start = int(line.split(' ')[0].split('\t')[1])
            trial_id = line.split(' ')[1].split('_')[2]
            # print("Line{}: {}".format(count, line.strip()))
        elif 'trials' in line and 'END' in line:
            trial_end = int(line.split(' ')[0].split('\t')[1])
            trial_length = int(trial_end) - int(trial_start)
            print(f'{trial_id} trials start: {trial_start-offset}, end: {trial_end-offset},  length: {trial_length}')

        if 'fc' in line and 'START' in line:
            fc_start = int(line.split(' ')[0].split('\t')[1])
        elif 'fc' in line and 'END' in line:
            fc_end = int(line.split(' ')[0].split('\t')[1])
            fc_length = int(fc_end) - int(fc_start)
            print(f'{trial_id} fc start: {fc_start-offset}, end: {fc_end-offset}, length: {fc_length}')

        if 'left_cue' in line and 'START' in line:
            left_cue_start = int(line.split(' ')[0].split('\t')[1])
        elif 'left_cue' in line and 'END' in line:
            left_cue_end = int(line.split(' ')[0].split('\t')[1])
            left_cue_length = int(left_cue_end) - int(left_cue_start)
            print(f'{trial_id} left_cue start: {left_cue_start-offset}, end: {left_cue_end-offset}, length: {left_cue_length}')

        if 'stim' in line and 'START' in line:
            stim_start = int(line.split(' ')[0].split('\t')[1])
        elif 'stim' in line and 'END' in line:
            stim_end = int(line.split(' ')[0].split('\t')[1])
            stim_length = int(stim_end) - int(stim_start)
            print(f'{trial_id} stim start: {stim_start-offset}, end: {stim_end-offset}, length: {stim_length}')