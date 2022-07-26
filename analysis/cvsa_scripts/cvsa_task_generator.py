#!/usr/bin/env python
"""
This script generates experiment settings files for nfblab based on participant config files
If appropriate data is present, it calculates all relevent thresholds and calibrations from files specified in <conf>.ini
It then generates the rest of the experiment scripts
"""

from experiment_generators.participant_generator import ParticipantTaskGenerator
from iaf_calculation import iaf_from_baseline
from eye_calibration import eye_calibration
from cvsa_thresholding import cvsa_threshold, cvsa_threshold_bv
import platform
import configparser
import os
import sys

if __name__ == "__main__":
    # read in the config
    config = configparser.ConfigParser()
    # config.read('example.ini')
    config.read(sys.argv[1])

    # if data is present, calculate the appropriate thresholds and generate the rest of the experiment files
    iaf = True
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
        eye_centre, eye_range = eye_calibration(config['FILES']['eye_calibration'])
        eye_threshold = True
    eye_centre = 0
    eye_range = 500
    eye_threshold = True

    aai_thresholds = False
    aai_threshold_mean = 0
    aai_threshold_max = 1
    if config['FILES']['aai_test']:
        # GENERATE the AAI thresholds
        aai_thresholds = True
        if config['FILES']['aai_test'].endswith('vhdr'):
            mu, std = cvsa_threshold_bv(config['FILES']['aai_test'], plot=True, alpha_band=(band_low, band_high))
        else:
            mu, std = cvsa_threshold(config['FILES']['aai_test'], plot=True, alpha_band=(band_low, band_high))
        aai_threshold_max = 2 * std
        threshold_extra = 0.25 * (aai_threshold_max - mu)
        aai_threshold_mean = mu + threshold_extra

    nfb_template = None
    if eye_threshold and aai_thresholds and iaf:
        nfb_template = "cvsa_feedback.xml"

    if platform.system() == "Windows":
        userdir = os.path.join("2354158T", "OneDrive - University of Glasgow")
        # mock_file_path = f'/Users/{userdir}/Documents/cvsa_pilot_testing/lab_test_20220428/0-nfb_task_ct_test_04-28_17-39-22/experiment_data.h5'
        mock_file_path = f'/Users/Chris/Documents/mock_data/0-baseline_ct_test_04-28_16-49-26/experiment_data.h5'
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
    nfb_duration = 15

    # Generate the settings for each session
    # NOTE!!: don't forget to freeze these once generated (so as to not loose randomisation
    tasks = {"baseline": "baseline.xml",
             "eye_calibration": "eye_calibration.xml",
             "nfb_task": nfb_template,
             "posner_task": "posner_psychopy.xml"}

    task_info = {}

    left_spatial_filter_scalp = ""
    right_spatial_filter_scalp = ""
    source_fb = False
    posner_test = 0
    show_score = 0
    enable_posner = 0
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
                    number_nfb_tasks = 5
                    posner_test = 0
                    stim_duration = 1.5
                    nfb_duration = 10
                    show_score = 0
                    enable_posner = 0
                if task == "posner_task":
                    number_nfb_tasks = 75
                    posner_test = 1
                    stim_duration = 1
                    nfb_duration = 8
                    show_score = 0
                    enable_posner = 1
                elif task == "nfb_task":
                    number_nfb_tasks = 20
                    posner_test = 0
                    stim_duration = 10
                    nfb_duration = 15
                    show_score = 0
                    enable_posner = 0


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
                                               use_aai_threshold=use_aai_threshold,
                                               nfb_duration=nfb_duration,
                                               show_score=show_score,
                                               enable_posner=enable_posner)
                Tsk.create_task(participant=participant_no)
