from collections import namedtuple, OrderedDict
import mne
import numpy as np
from matplotlib import pyplot as plt


Channel = namedtuple('Channel', 'pos type')

class Montage:

    CH_TYPE_OTHER = 'Other'

    def __init__(self, ch_ordered_dict):
        '''
        :param ch_dict: OrderedDict of Channel instances with channel names as keys
        '''
        assert (isinstance(ch_ordered_dict, OrderedDict)
                and len(ch_ordered_dict) != 0
                and all(isinstance(ch_ordered_dict[ch], Channel) for ch in ch_ordered_dict)
                )
        self.ch_dict = ch_ordered_dict
        self.ch_types = set([chan.type for chan in self.ch_dict.values()])

    @classmethod
    def create_from_names(cls, names, file_path=None):

        layout_1005 = mne.channels.read_layout('EEG1005')
        layout_mag = mne.channels.read_layout('Vectorview-mag')
        layout_grad = mne.channels.read_layout('Vectorview-grad')

        user_ch_dict = cls.__read_layout_from_file(file_path)

        ch_dict = OrderedDict()
        for name in names:
            if name in user_ch_dict.keys():
                ch_dict[name] = user_ch_dict[name]
            elif name in layout_1005.names:
                ch_ind = layout_1005.names.index(name)
                ch_dict[name] = Channel(layout_1005.pos[ch_ind][:2], 'EEG')
            elif name in layout_mag.names:
                ch_ind = layout_mag.names.index(name)
                ch_dict[name] = Channel(layout_mag.pos[ch_ind][:2], 'MAG')
            elif name in layout_grad.names:
                ch_ind = layout_grad.names.index(name)
                ch_dict[name] = Channel(layout_grad.pos[ch_ind][:2], 'GRAD')
            else:
                ch_dict[name] = Channel(None, cls.CH_TYPE_OTHER)

        return cls(ch_dict)

    @classmethod
    def __read_layout_from_file(cls, file_path):
        if file_path is None:
            return OrderedDict()

    def pick_channels_by_type(self, ch_type):
        return np.nonzero([self.ch_dict[ch].type == ch_type for ch in self.ch_dict])[0]

    def plot_topography(self, data, types, axs=None):
        '''Plots topographies. Duh..

        :param data: vector of numbers for each channel (including 'Aux' and similar)
        :param types: list of types of channels to plot. has to be a subset of types in the montage.
        :param axs: optional list of axes objects to plot into. if supplied should be the same lengths as types
        :return: list of axes objects it has plotted into
        '''

        plottable_types = self.ch_types.difference([self.CH_TYPE_OTHER])
        assert plottable_types.issuperset(types), "Can't plot channel types {}".format(
            str(set(types).difference(plottable_types)))
        assert len(types) != 0, 'Supply at least one channel type for plotting'
        assert axs is None or len(types) == len(axs), "Numbers of channel types and axes objects don't match"

        if axs is None:
            fig, axs = plt.subplots(1, len(types))
            if len(types) == 1:
                axs = [axs]

        ch_list = list(self.ch_dict.values())
        for ch_type, axes in zip(types, axs):
            axes.set_title(ch_type)
            ch_inds = self.pick_channels_by_type(ch_type)
            pos = np.asarray([ch_list[ch_ind].pos for ch_ind in ch_inds])
            type_data = np.asarray(data)[ch_inds]
            mne.viz.plot_topomap(type_data, pos=pos, axes=axes, show=False)


        return axs





if __name__ == '__main__':
    layout_1005 = mne.channels.read_layout('EEG1005')
    layout_mag = mne.channels.read_layout('Vectorview-mag')
    layout_grad = mne.channels.read_layout('Vectorview-grad')
    ch_list_from_lsl = layout_1005.names + layout_mag.names + layout_grad.names + ['AUX1']
    print('The following channels names were received from the LSL stream:')
    print(ch_list_from_lsl)
    print()

    print('And here are the position and types that were assigned to them')
    montage = Montage.create_from_names(ch_list_from_lsl)
    for name in montage.ch_dict.keys():
        print(name)
        print(montage.ch_dict[name].pos)
        print(montage.ch_dict[name].type)
        print()

    types_set = montage.ch_types.difference([montage.CH_TYPE_OTHER])
    print('The montage includes the following types (excluding {}):'.format(montage.CH_TYPE_OTHER))
    print(types_set)
    print()

    data = np.random.random(len(ch_list_from_lsl))

    axs = montage.plot_topography(data, types_set)
    plt.show()

    axs2 = montage.plot_topography(data, [list(types_set)[1]])
    plt.show()

    # Throws an error
    axs2 = montage.plot_topography(data, ['Aux', 'EEG'])
    plt.show()
