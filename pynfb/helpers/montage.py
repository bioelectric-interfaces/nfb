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
        layout_neuromag = read_layout(kind='Vectorview-all')
        layout_eeg1020 = read_layout(kind='EEG1005')
        if file_path is not None:
            layout_user = cls.read_layout_from_file(file_path)
        ch_list = []
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
from pynfb.helpers.az_proj import azimuthal_equidistant_projection as azim
# pickled mne.Info instance from a random neuromag file
fpath = os.path.join(os.getcwd(), "pynfb/helpers/neuromag_info.p")
info = pickle.load(open(fpath, 'rb'))

neuromag_ch_list = [Channel(ch['ch_name'], ch['loc'][:3], mne.io.pick.channel_type(info, idx))
                         for (idx, ch) in enumerate(info['chs'])]
neuromag_ch_list = [Channel(ch.name, azim(ch.pos[None, :]), ch.type) for ch in neuromag_ch_list if ch.type in ['grad', 'mag']]

montage_1005 = mne.channels.read_montage('standard_1005')
montage_1020 = mne.channels.read_montage('standard_1020')

ch_list_1005 = [Channel(name, azim(pos[None, :]), 'EEG') for (name, pos) in zip(montage_1005.ch_names, montage_1005.pos)]
ch_list_1020 = [Channel(name, azim(pos[None, :]), 'EEG') for (name, pos) in zip(montage_1020.ch_names, montage_1020.pos)]
