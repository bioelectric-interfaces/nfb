#!/usr/bin/env python

# TODO: make signal and protocol generic blocks for the template

import os
from jinja2 import Environment, FileSystemLoader

import image_list_generator as ilg

PATH = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_ENVIRONMENT = Environment(
    autoescape=False,
    loader=FileSystemLoader(os.path.join(PATH, 'templates')),
    trim_blocks=False)


class ParticipantTaskGenerator:

    def __init__(self, template_file="freeview_template.xml", experiment_prefix="task", participant_no=999,
                 stream_name="eeg_bci_test",
                 image_path="", band_low=8, band_high=12, t_filt_type='fft', composite_signal="AAI", free_view_images=None,
                 number_nfb_tasks=5, baseline_duration=3):
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

    def render_template(self, template_filename, context):
        return TEMPLATE_ENVIRONMENT.get_template(template_filename).render(context)

    def create_task(self):
        output_fname = f"{self.experiment_prefix}_{self.participant_no}.xml"

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
            'baseline_duration': self.baseline_duration
        }
        #
        with open(output_fname, 'w') as f:
            output = self.render_template(self.template_file, context)
            f.write(output)

########################################

if __name__ == "__main__":
    # TODO:
    #   [ ] make this runnable from command line
    #   [ ] make participant number the argument to pass
    #   [ ] make scalp and source sets (pre/nfb/post tasks)
    #       [ ]make sure the order is randomised and number the sets _1, _2, _3
    #   [ ] make sham set as above
    # Common settings
    iamges_path = '/Users/christopherturner/Documents/ExperimentImageSet'
    participant_no = 999
    stream_name = "eeg_bci_test"
    image_path = ""
    band_low = 8
    band_high = 12
    t_filt_type = 'fft'
    composite_signal = "AAI"

    ImgGen = ilg.ImageListGenerator(iamges_path)
    pre_task_images, post_task_images = ImgGen.get_pre_post_images()
    # pre-task settings
    pre_task_prefix = "pre-task"
    pre_template = "freeview_template.xml"

    # post-task settings
    post_task_prefix = "post-task"
    post_template = pre_template

    # NFB task settings
    nfb_task_prefix = "nfb-task"
    nfb_template = "nfb_template.xml"

    PreTask = ParticipantTaskGenerator(participant_no=participant_no,
                                       stream_name=stream_name,
                                       image_path=iamges_path,
                                       band_low=band_low,
                                       band_high=band_high,
                                       t_filt_type=t_filt_type,
                                       composite_signal=composite_signal,
                                       experiment_prefix=pre_task_prefix,
                                       template_file=pre_template,
                                       free_view_images=pre_task_images)

    PostTask = ParticipantTaskGenerator(participant_no=participant_no,
                                        stream_name=stream_name,
                                        image_path=iamges_path,
                                        band_low=band_low,
                                        band_high=band_high,
                                        t_filt_type=t_filt_type,
                                        composite_signal=composite_signal,
                                        experiment_prefix=post_task_prefix,
                                        template_file=post_template,
                                        free_view_images=post_task_images)

    NFBTask = ParticipantTaskGenerator(participant_no=participant_no,
                                       stream_name=stream_name,
                                       image_path=iamges_path,
                                       band_low=band_low,
                                       band_high=band_high,
                                       t_filt_type=t_filt_type,
                                       composite_signal=composite_signal,
                                       experiment_prefix=nfb_task_prefix,
                                       template_file=nfb_template)
    PreTask.create_task()
    PostTask.create_task()
    NFBTask.create_task()

