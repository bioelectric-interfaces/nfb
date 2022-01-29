#!/usr/bin/env python

# TODO: make signal and protocol generic blocks for the template

import os
import random
from jinja2 import Environment, FileSystemLoader

import image_list_generator as ilg

PATH = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_ENVIRONMENT = Environment(
    autoescape=False,
    loader=FileSystemLoader(os.path.join(PATH, 'templates')),
    trim_blocks=False)


class ParticipantTaskGenerator:

    # TODO: make a config object to simplify this init
    def __init__(self, template_file="freeview_template.xml", experiment_prefix="task", participant_no="999",
                 stream_name="eeg_bci_test",
                 image_path="", band_low=8, band_high=12, t_filt_type='fft', composite_signal="AAI",
                 free_view_images=None,
                 number_nfb_tasks=5, baseline_duration=3, left_spatial_filter_scalp="P4=1",
                 right_spatial_filter_scalp="P3=1",
                 source_fb=False, source_roi_left=(), source_roi_right=(), mock_file=None,
                 baseline_cor_threshold=0.25, use_baseline_correction=1, enable_smoothing=1, smooth_window=100,
                 fft_window=250, mock_reward_threshold=0.0):

        self.template_file = template_file
        self.composite_signal = composite_signal
        self.band_high = band_high
        self.band_low = band_low
        self.image_path = image_path
        self.stream_name = stream_name
        self.participant_no = participant_no
        self.t_filt_type = t_filt_type
        self.experiment_prefix = experiment_prefix
        self.free_view_images = free_view_images
        self.number_nfb_tasks = number_nfb_tasks
        self.feedback_display = {}
        self.baseline_duration = baseline_duration
        self.source_fb = source_fb
        if source_fb:
            self.left_spatial_filter_scalp = ""
            self.right_spatial_filter_scalp = ""
        else:
            self.left_spatial_filter_scalp = left_spatial_filter_scalp
            self.right_spatial_filter_scalp = right_spatial_filter_scalp
        self.source_roi_left = source_roi_left
        self.source_roi_right = source_roi_right
        self.mock_file = mock_file
        self.use_baseline_correction = use_baseline_correction
        self.baseline_cor_threshold = baseline_cor_threshold
        self.smooth_window = smooth_window
        self.enable_smoothing = enable_smoothing
        self.fft_window = fft_window
        self.mock_reward_threshold = mock_reward_threshold

    def render_template(self, template_filename, context):
        return TEMPLATE_ENVIRONMENT.get_template(template_filename).render(context)

    def create_task(self, participant=999):
        output_fname = f"{self.experiment_prefix}_{self.participant_no}.xml"
        output_dir = os.path.join("experiment_config_output", str(participant))
        if not os.path.exists(output_dir):
            # Create a new directory because it does not exist
            os.makedirs(output_dir)
        output_fname = os.path.join(output_dir, output_fname)

        # Todo make this a class member and init in init
        context = {
            'experiment': f"{self.experiment_prefix}_{self.participant_no}",
            'stream_name': self.stream_name,
            'image_set': self.free_view_images,
            'temp_filt_type': self.t_filt_type,
            'band_low': self.band_low,
            'band_high': self.band_high,
            'composite_signal': self.composite_signal,
            'number_nfb_tasks': self.number_nfb_tasks,
            'fb_display': self.feedback_display,
            'baseline_duration': self.baseline_duration,
            'right_spatial_filter_scalp': self.right_spatial_filter_scalp,
            'left_spatial_filter_scalp': self.left_spatial_filter_scalp,
            'source_roi_left': self.source_roi_left,
            'source_roi_right': self.source_roi_right,
            'source_fb': int(self.source_fb),
            'mock_file': self.mock_file,
            'use_baseline_correction': self.use_baseline_correction,
            'baseline_cor_threshold': self.baseline_cor_threshold,
            'smooth_window': self.smooth_window,
            'enable_smoothing': self.enable_smoothing,
            'fft_window': self.fft_window,
            'mock_reward_threshold': self.mock_reward_threshold
        }
        #
        with open(output_fname, 'w') as f:
            output = self.render_template(self.template_file, context)
            f.write(output)


########################################

if __name__ == "__main__":
    # TODO:
    #   [ ] make sure the tasks are exactly how they should be (having choice, delays, correct text, etc)
    #   [ ] make this runnable from command line
    #       [ ] make participant number the argument to pass
    #   [ ] sham set as above
    #   [ ] make sure images are only shown once (so separate folder for sham, scalp, source)

    # TODO: generate a participant directory with scalp, source, sham subdirs in USERS/HOME/EEG_DATA

    # Base images path
    # base_images_path = "image_stimuli"
    # base_images_path = "/Users/2354158T/Documents/image_stimuli"
    base_images_path = "/Users/2354158T/OneDrive - University of Glasgow/Documents/PilotImageSet_forExperiment"
    # base_images_path = "/Users/christopherturner/Documents/ExperimentImageSet/bagherzadeh/image_stimuli"
    session_images_paths = [os.path.join(base_images_path, x) for x in ["0", "1", "2"]] # 0 = scalp, 1 = source, 2 = sham # TODO: MAKE SURE THIS IS RANDOMISED

    # randomise session order
    random.shuffle(session_images_paths)

    # Common settings
    participant_no = "kk"
    stream_name = "BrainVision RDA"
    image_path = ""
    band_low = 8
    band_high = 12
    t_filt_type = 'fft'
    composite_signal = "AAI"
    baseline_duration = 120
    number_nfb_tasks = 100
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
             "nfb_task": "nfb_template.xml"}

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
            left_spatial_filter_scalp = "CP5=1;P5=1;O1=1"
            right_spatial_filter_scalp = "CP6=1;P6=1;O2=1"
        elif session == 1:
            # source
            source_roi_left = ("inferiorparietal-lh", "superiorparietal-lh", "lateraloccipital-lh")
            source_roi_right = ("inferiorparietal-rh", "superiorparietal-rh", "lateraloccipital-rh")
            source_fb = True
        elif session == 2:
            # sham
            left_spatial_filter_scalp = "CP5=1;P5=1;01=1"
            right_spatial_filter_scalp = "CP6=1;P6=1;02=1"
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
                                           mock_reward_threshold=mock_reward_threshold)
            Tsk.create_task(participant=participant_no)
