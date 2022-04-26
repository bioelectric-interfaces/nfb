
import os
from jinja2 import Environment, FileSystemLoader


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
                 baseline_cor_threshold=0.25, use_baseline_correction=1, enable_smoothing=0, smooth_window=100,
                 fft_window=250, mock_reward_threshold=0.0, nfb_type=2, posner_test=0, eye_range=500, eye_threshold=1, stim_duration=3, muscle_signal=''):

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
        self.nfb_type = nfb_type
        self.posner_test = posner_test
        self.stim_duration = stim_duration
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
        self.eye_range = eye_range
        self.eye_threshold = eye_threshold
        self.muscle_signal = muscle_signal

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
            'mock_reward_threshold': self.mock_reward_threshold,
            'nfb_type': self.nfb_type,
            'posner_test': self.posner_test,
            'eye_range': self.eye_range,
            'eye_threshold': self.eye_threshold,
            'stim_duration': self.stim_duration,
            'muscle_signal': self.muscle_signal
        }
        #
        with open(output_fname, 'w') as f:
            output = self.render_template(self.template_file, context)
            f.write(output)
