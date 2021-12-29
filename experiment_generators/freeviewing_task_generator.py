#!/usr/bin/env python

import os
from jinja2 import Environment, FileSystemLoader

import image_list_generator as ilg

PATH = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_ENVIRONMENT = Environment(
    autoescape=False,
    loader=FileSystemLoader(os.path.join(PATH, 'templates')),
    trim_blocks=False)

PARTICIPANT_NO = 999
STREAM_NAME = "eeg_bci_test"
IMAGE_PATH = '/Users/christopherturner/Documents/ExperimentImageSet'
BAND_LOW = 8
BAND_HIGH = 12
TEMPORAL_FILTER_TYPE = "fft"
COMPOSITE_SIGNAL = 'AAI'


def render_template(template_filename, context):
    return TEMPLATE_ENVIRONMENT.get_template(template_filename).render(context)


def create_index_html():
    pre_fname = f"pre-task_{PARTICIPANT_NO}.xml"
    post_fname = f"post-task_{PARTICIPANT_NO}.xml"

    ImgGen = ilg.ImageListGenerator(IMAGE_PATH)
    pre_task_images, post_task_images = ImgGen.get_pre_post_images()
    pre_context = {
        'experiment': f"pre-task_{PARTICIPANT_NO}",
        'stream_name': STREAM_NAME,
        'image_set': pre_task_images,
        'temp_filt_type': TEMPORAL_FILTER_TYPE,
        'band_low': BAND_LOW,
        'band_high': BAND_HIGH,
        'composite_signal': COMPOSITE_SIGNAL
    }
    #
    with open(pre_fname, 'w') as f:
        html = render_template('freeview_template.xml', pre_context)
        f.write(html)

    post_context = {
        'experiment': f"pre-task_{PARTICIPANT_NO}",
        'stream_name': "eeg_bci_test",
        'image_set': post_task_images
    }
    #
    with open(post_fname, 'w') as f:
        html = render_template('freeview_template.xml', post_context)
        f.write(html)


def main():
    create_index_html()


########################################

if __name__ == "__main__":
    main()