#!/usr/bin/env python

import os
from jinja2 import Environment, FileSystemLoader

import image_list_generator as ilg

PATH = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_ENVIRONMENT = Environment(
    autoescape=False,
    loader=FileSystemLoader(os.path.join(PATH, 'templates')),
    trim_blocks=False)


def render_template(template_filename, context):
    return TEMPLATE_ENVIRONMENT.get_template(template_filename).render(context)


def create_index_html():
    participant_no = 999
    pre_fname = f"pre-task_{participant_no}.xml"
    post_fname = f"post-task_{participant_no}.xml"

    ImgGen = ilg.ImageListGenerator('/Users/christopherturner/Documents/ExperimentImageSet')
    pre_task_images, post_task_images = ImgGen.get_pre_post_images()
    # TODO: add signal bit
    pre_context = {
        'experiment': f"pre-task_{participant_no}",
        'stream_name': "eeg_bci_test",
        'image_set': pre_task_images
    }
    #
    with open(pre_fname, 'w') as f:
        html = render_template('freeview_template.xml', pre_context)
        f.write(html)

    post_context = {
        'experiment': f"pre-task_{participant_no}",
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