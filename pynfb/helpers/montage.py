from pynfb.widgets.helpers import ch_names_to_2d_pos
from collections import namedtuple
from mne.channels import read_montage
from mne.channels import read_layout
from mne import pick_channels


Channel = namedtuple('Channel', 'name pos type')

class Montage:
    def __init__(self, ch_list):
        '''
        :param ch_list: list of Channel instances
        '''
        assert isinstance(ch_list, list) and len(ch_list) != 0 and all(isinstance(ch, Channel) for ch in ch_list)
        self.ch_list = ch_list

    @classmethod
    def create_from_names(cls, names, file_path=None):
        # put your code here ***********
        # you can use "ch_names_to_2d_pos()"
        # use file_path if not None
        layout_1005 = mne.channels.read_layout('EEG1005')
        layout_mag = mne.channels.read_layout('Vectorview-mag')
        layout_grad = mne.channels.read_layout('Vectorview-grad')
        if file_path is not None:
            user_ch_list = cls.read_layout_from_file(file_path)
        for name in names:

        return cls(ch_list)

    def pick_types(self, type):
        return [ch.type == type for ch in ch_list]

    def plot_topography(self, data, types, axes=None):
        assert len(types) == len(axes) or axes is None
        return axes





if __name__ == '__main__':
    ch_info_from_lsl = ['Fp1', 'Fp2', 'Cz']

ch1 = Channel('a', (0, 0, 0), 'EEG')
ch2 = Channel('b', (0, 0, 0), 'ECG')
ch3 = Channel('c', (0, 0, 0), 'EEG')
ch_list = [ch1, ch2, ch3]



import pickle
import mne




ch_list_1005 = [Channel(name, pos[:2], 'EEG') for name, pos in zip(layout_1005.names, layout_1005.pos)]
