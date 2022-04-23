import numpy as np
import mne
import pandas as pd
import packaging.version


def azimuthal_equidistant_projection(hsp):
    radius = 20
    width = 5
    height = 4
    # Calculate angles
    r = np.sqrt(np.sum(hsp ** 2, axis=-1))
    theta = np.arccos(hsp[:, 2] / r)
    phi = np.arctan2(hsp[:, 1], hsp[:, 0])

    # Mark the points that might have caused bad angle estimates
    iffy = np.nonzero(np.sum(hsp[:, :2] ** 2, axis=-1) ** (1. / 2)
                      < np.finfo(np.float).eps * 10)
    theta[iffy] = 0
    phi[iffy] = 0

    # Do the azimuthal equidistant projection
    x = radius * (2.0 * theta / np.pi) * np.cos(phi)
    y = radius * (2.0 * theta / np.pi) * np.sin(phi)

    pos = np.c_[x, y]
    return pos


class Montage(pd.DataFrame):
    CHANNEL_TYPES = ['EEG', 'MAG', 'GRAD', 'OTHER']

    def __init__(self, names, **kwargs):
        if not isinstance(names, list):
            super(Montage, self).__init__(names, **kwargs)
        else:
            super(Montage, self).__init__(columns=['name', 'type', 'pos_x', 'pos_y'])
            layout_eeg = Montage.load_layout('EEG1005')
            layout_mag = Montage.load_layout('Vectorview-mag')
            layout_mag.names = list(map(lambda x: x.replace(' ', ''), layout_mag.names))
            layout_grad = Montage.load_layout('Vectorview-grad')
            layout_grad.names = list(map(lambda x: x.replace(' ', ''), layout_grad.names))
            for name in names:
                if name.upper() in layout_eeg.names:
                    ch_ind = layout_eeg.names.index(name.upper())
                    self._add_channel(name, 'EEG', layout_eeg.pos[ch_ind][:2])
                elif name.upper() in layout_mag.names:
                    ch_ind = layout_mag.names.index(name.upper())
                    self._add_channel(name, 'MAG', layout_mag.pos[ch_ind][:2])
                elif name.upper() in layout_grad.names:
                    ch_ind = layout_grad.names.index(name.upper())
                    self._add_channel(name, 'GRAD', layout_grad.pos[ch_ind][:2])
                else:
                    self._add_channel(name, 'OTHER', (None, None))

    def _add_channel(self, name, type, pos):
        self.loc[len(self)] = {'name': name, 'type': type, 'pos_x': pos[0], 'pos_y': pos[1]}

    def get_names(self, type='ALL'):
        return list(self[self.get_mask(type)]['name'])

    def get_pos(self, type='ALL'):
        return (self[self.get_mask(type)][['pos_x', 'pos_y']]).values

    def get_mask(self, type='ALL'):
        if type in self.CHANNEL_TYPES:
            return (self['type'] == type).values
        elif type == 'ALL':
            return (self['type'] == self['type']).values
        elif type == 'GRAD2':
            return ((self['type'] == 'GRAD') & (self['name'].apply(lambda x: x[-1]) == '2')).values
        elif type == 'GRAD3':
            return ((self['type'] == 'GRAD') & (self['name'].apply(lambda x: x[-1]) == '3')).values
        else:
            raise TypeError('Bad channels type')

    @staticmethod
    def load_layout(name):
        if name == 'EEG1005':
            if packaging.version.parse(mne.__version__) >= packaging.version.parse("0.19"):  # validate mne version (mne 0.19+)
                layout = mne.channels.make_standard_montage('standard_1005')
                layout.names = layout.ch_names
                ch_pos_dict = layout._get_ch_pos()
                layout.pos = np.array([ch_pos_dict[name] for name in layout.names])
                layout.pos = azimuthal_equidistant_projection(layout.pos)
            else:
                layout = mne.channels.read_montage('standard_1005')
                layout.pos = azimuthal_equidistant_projection(layout.pos)
                layout.names = layout.ch_names
        else:
            layout = mne.channels.read_layout(name)
        layout.names = list(map(str.upper, layout.names))
        return layout

    def make_laplacian_proj(self, type='ALL', n_channels=4):
        pos = self.get_pos(type)
        proj = np.eye(pos.shape[0])
        for k in range(pos.shape[0]):
            proj[k, np.argsort(((pos[k] - pos) ** 2).sum(1))[1:1+n_channels]] = -1 / n_channels
        return proj

    def combine_grad_data(self, data):
        names = self.get_names('GRAD')
        grad2_mask = list(map(lambda x: x[-1]=='2', names))
        grad3_mask = list(map(lambda x: x[-1] == '3', names))
        combined_data = (data[grad2_mask]**2 + data[grad3_mask]**2)**0.5
        return combined_data, self.get_pos('GRAD')[grad2_mask]

if __name__ == '__main__':
    m = Montage(['cz', 'fp1', 'FP2', 'AUX1', 'MEG2632', 'MEg2633', 'MEG2332', 'MEg2333', 'Pz', 'Fcz'])
    print(m)
    print(m.get_names('EEG'))
    print(m.get_pos('EEG'))
    print(len(m))
    print(m.get_mask('EEG'))
    print(m.make_laplacian_proj('EEG'))
    print(m.get_mask('GRAD2'))

    print(m.combine_grad_data(np.arange(len(m.get_names('GRAD')))))