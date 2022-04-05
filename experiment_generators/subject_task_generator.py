#!/usr/bin/env python

# TODO: make signal and protocol generic blocks for the template
import os
import random
from participant_generator import ParticipantTaskGenerator
import platform
import image_list_generator as ilg
########################################

if __name__ == "__main__":
    # TODO:
    #   [ ] make sure the tasks are exactly how they should be (having choice, delays, correct text, etc)
    #   [ ] make this runnable from command line
    #       [ ] make participant number the argument to pass
    #   [ ] sham set as above
    #   [ ] make sure images are only shown once (so separate folder for sham, scalp, source)

    # TODO: generate a participant directory with scalp, source, sham subdirs in USERS/HOME/EEG_DATA

    nfb_types = {"circle": 1, "bar": 2, "gabor": 3, "plot": 4}
    if platform.system() == "Windows":
        userdir = "2354158T"
        base_images_path = "/Users/2354158T/OneDrive - University of Glasgow/Documents/PilotImageSet_forExperiment"
    else:
        userdir = "christopherturner"
        base_images_path = "/Users/christopherturner/Documents/ExperimentImageSet/bagherzadeh/image_stimuli"
    # Base images path
    # base_images_path = "image_stimuli"
    # base_images_path = "/Users/2354158T/Documents/image_stimuli"

    session_images_paths = [os.path.join(base_images_path, x) for x in ["0", "1", "2"]] # 0 = scalp, 1 = source, 2 = sham # TODO: MAKE SURE THIS IS RANDOMISED

    # randomise session order
    random.shuffle(session_images_paths)

    # Common settings
    participant_no = "tst_gabor2"
    stream_name = "BrainVision RDA"
    image_path = ""
    band_low = 8
    band_high = 12
    t_filt_type = 'fft'
    composite_signal = "AAI"
    baseline_duration = 90
    number_nfb_tasks = 10
    nfb_type = nfb_types['bar']
    # nfb_template = "nfb_template_graph.xml" #"nfb_template_gabor.xml"
    nfb_template = "nfb_template_gabor.xml"
    use_baseline_correction = 1
    baseline_cor_threshold = 0.2
    mock_file = ''
    smooth_window = 100 # THIS IS AAI SMOOTHING
    enable_smoothing = 1 # THIS IS AAI SMOOTHING
    fft_window = 1000
    mock_reward_threshold = 0.089

    # Generate the settings for each session
    # NOTE!!: don't forget to freeze these once generated (so as to not loose randomisation
    tasks = {"pre_task": "freeview_template.xml", "post_task": "freeview_template.xml",
             "nfb_task": nfb_template}

    for session, image_path in enumerate(session_images_paths):
        task_info = {}
        ImgGen = ilg.ImageListGenerator(image_path)
        pre_task_images, post_task_images = ImgGen.get_pre_post_images()

        left_spatial_filter_scalp = ""
        right_spatial_filter_scalp = ""
        source_roi_left = ()
        source_roi_right = ()
        source_fb = False
        if session == 0:
            # scalp
            left_spatial_filter_scalp = "PO7=1"#"PO7=1;P5=1;O1=1"
            right_spatial_filter_scalp = "PO8=1"#"PO8=1;P6=1;O2=1"
        elif session == 1:
            # source
            source_roi_left = ("inferiorparietal-lh", "superiorparietal-lh", "lateraloccipital-lh")
            source_roi_right = ("inferiorparietal-rh", "superiorparietal-rh", "lateraloccipital-rh")
            source_fb = True
        elif session == 2:
            # sham
            left_spatial_filter_scalp = "PO7=1;P5=1;01=1"
            right_spatial_filter_scalp = "PO8=1;P6=1;02=1"
            # mock_file = '/Users/christopherturner/Documents/EEG_Data/pilot_202201/sh/scalp/0-nfb_task_SH01_01-11_15-50-56/experiment_data.h5'
            mock_file = '/Users/2354158T/Documents/EEG_Data/pilot_202201_sham/0-nfb_task_ct02_01-26_16-33-42/experiment_data.h5'
        for task, template in tasks.items():
            free_view_images = None
            if task == "pre_task":
                free_view_images = pre_task_images
            elif task == "post_task":
                free_view_images = post_task_images
            Tsk = ParticipantTaskGenerator(participant_no=participant_no,
                                           stream_name=stream_name,
                                           image_path=image_path,
                                           band_low=band_low,
                                           band_high=band_high,
                                           t_filt_type=t_filt_type,
                                           composite_signal=composite_signal,
                                           experiment_prefix=f"{session}-{task}",
                                           template_file=template,
                                           free_view_images=free_view_images,
                                           baseline_duration=baseline_duration,
                                           right_spatial_filter_scalp=right_spatial_filter_scalp,
                                           left_spatial_filter_scalp=left_spatial_filter_scalp,
                                           source_roi_left=source_roi_left,
                                           source_roi_right=source_roi_right,
                                           source_fb=source_fb,
                                           number_nfb_tasks=number_nfb_tasks,
                                           mock_file=mock_file,
                                           baseline_cor_threshold=baseline_cor_threshold,
                                           use_baseline_correction=use_baseline_correction,
                                           smooth_window=smooth_window,
                                           enable_smoothing=enable_smoothing,
                                           fft_window=fft_window,
                                           mock_reward_threshold=mock_reward_threshold,
                                           nfb_type=nfb_type)
            Tsk.create_task(participant=participant_no)
