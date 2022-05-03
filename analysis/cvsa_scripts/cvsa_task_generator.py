#!/usr/bin/env python
"""
This script generates experiment settings files for nfblab based on participant config files
If appropriate data is present, it calculates all relevent thresholds and calibrations from files specified in <conf>.ini
It then generates the rest of the experiment scripts
"""

from experiment_generators.participant_generator import ParticipantTaskGenerator
from iaf_calculation import iaf_from_baseline
import platform
import configparser

if __name__ == "__main__":
    # read in the config
    config = configparser.ConfigParser()
    config.read('example.ini')

    # if data is present, calculate the appropriate thresholds and generate the rest of the experiment files
    iaf = False
    if config['FILES']['baseline']:
        # GENERATE the IAF
        iaf = True
        _, iaf_ec = iaf_from_baseline(config['FILES']['baseline'])
        band_low = iaf_ec - 2
        band_high = iaf_ec + 2
    else:
        band_low = 8
        band_high = 12

    eye_threshold = False
    eye_range = 500
    if config['FILES']['eye_calibration']:
        # GENERATE the eye thresholds
        eye_threshold = True
        eye_range = 500

    aai_thresholds = False
    aai_threshold_mean = 0
    aai_threshold_max = 1
    if config['FILES']['aai_test']:
        aai_thresholds = True
        # GENERATE the AAI thresholds
        aai_threshold_mean = 0.2
        aai_threshold_max = 0.5

    nfb_template = None
    if eye_threshold and aai_thresholds and iaf:
        nfb_template = "cvsa_feedback.xml"

    if platform.system() == "Windows":
        userdir = "2354158T"
        mock_file_path = f'/Users/{userdir}/Documents/EEG_Data/pilot_202201_sham/0-nfb_task_ct02_01-26_16-33-42/experiment_data.h5'
    else:
        userdir = "christopherturner"
        mock_file_path = f'/Users/{userdir}/Documents/EEG_Data/pilot_202201/ct02/scalp/0-nfb_task_ct02_01-26_16-33-42/experiment_data.h5'
    nfb_types = {"circle": 1, "bar": 2, "gabor": 3, "plot": 4, "posner":5}

    # Common settings
    participant_no = config['EXPERIMENT']['participant_id']
    stream_name = "BrainVision RDA"
    t_filt_type = 'fft'
    composite_signal = "AAI"
    number_nfb_tasks = 10
    nfb_type = nfb_types['posner']
    # nfb_template = "nfb_template_graph.xml" #"nfb_template_gabor.xml"
    test_template = "cvsa_feedback.xml"
    use_baseline_correction = 0
    baseline_cor_threshold = 0.2
    smooth_window = 100 # THIS IS AAI SMOOTHING
    enable_smoothing = 1 # THIS IS AAI SMOOTHING
    fft_window = 1000
    mock_reward_threshold = 0.089
    eye_threshold = int(eye_threshold)
    eye_range = eye_range
    stim_duration = 5 # TODO - make this random?
    baseline_duration = 120
    use_aai_threshold = int(aai_thresholds)

    # Generate the settings for each session
    # NOTE!!: don't forget to freeze these once generated (so as to not loose randomisation
    tasks = {"baseline": "baseline.xml",
             "eye_calibration": "eye_calibration.xml",
             "test_task": test_template,
             "nfb_task": nfb_template}

    task_info = {}

    left_spatial_filter_scalp = ""
    right_spatial_filter_scalp = ""
    source_fb = False
    posner_test = 0
    for session in [0, 1]:
        if session == 0:
            # scalp
            left_spatial_filter_scalp = "PO7=1"#"PO7=1;P5=1;O1=1"
            right_spatial_filter_scalp = "PO8=1"#"PO8=1;P6=1;O2=1"
            mock_file = ''
            muscle_signal = 'EYE_TRACK'
        elif session == 1:
            # sham
            left_spatial_filter_scalp = "PO7=1"#;P5=1;01=1"
            right_spatial_filter_scalp = "PO8=1"#;P6=1;02=1"
            mock_file = mock_file_path
            muscle_signal = ''
        for task, template in tasks.items():
            if template:
                if task == "test_task":
                    number_nfb_tasks = 10
                    posner_test = 1
                    stim_duration = 3
                elif task == "nfb_task":
                    number_nfb_tasks = 25
                    posner_test = 0
                    stim_duration = 5


                Tsk = ParticipantTaskGenerator(participant_no=participant_no,
                                               stream_name=stream_name,
                                               band_low=band_low,
                                               band_high=band_high,
                                               t_filt_type=t_filt_type,
                                               composite_signal=composite_signal,
                                               experiment_prefix=f"{session}-{task}",
                                               template_file=template,
                                               right_spatial_filter_scalp=right_spatial_filter_scalp,
                                               left_spatial_filter_scalp=left_spatial_filter_scalp,
                                               source_fb=source_fb,
                                               number_nfb_tasks=number_nfb_tasks,
                                               mock_file=mock_file,
                                               baseline_cor_threshold=baseline_cor_threshold,
                                               use_baseline_correction=use_baseline_correction,
                                               smooth_window=smooth_window,
                                               enable_smoothing=enable_smoothing,
                                               fft_window=fft_window,
                                               mock_reward_threshold=mock_reward_threshold,
                                               nfb_type=nfb_type,
                                               posner_test=posner_test,
                                               eye_range=eye_range,
                                               eye_threshold=eye_threshold,
                                               stim_duration=stim_duration,
                                               baseline_duration=baseline_duration,
                                               muscle_signal=muscle_signal,
                                               aai_threshold_max=aai_threshold_max,
                                               aai_threshold_mean=aai_threshold_mean,
                                               use_aai_threshold=use_aai_threshold)
                Tsk.create_task(participant=participant_no)
