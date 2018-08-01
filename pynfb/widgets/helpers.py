from PyQt5 import QtCore, QtGui, QtWidgets

from ..helpers.az_proj import azimuthal_equidistant_projection

try:
    from mne.channels import read_montage
    from mne import pick_channels
except ImportError:
    pass
from numpy import array, random

DEBUG = False


def seems_to_come_from_neuromag(list_of_ch_names):
    return list_of_ch_names[0].startswith('MEG')


def ch_names_to_2d_pos(list_of_ch_names, kind='standard_1005', azimuthal=True):
    montage = read_montage(kind)
    if DEBUG or seems_to_come_from_neuromag(list_of_ch_names):
        return random.normal(size=(len(list_of_ch_names), 2))
    upper_list_of_ch_names = [ch.upper() for ch in list_of_ch_names]
    upper_montage_ch_names = [ch.upper() for ch in montage.ch_names]
    indices = [upper_montage_ch_names.index(ch) if ch in upper_montage_ch_names else 0 for ch in upper_list_of_ch_names]
    if len(list(indices)) < len(list_of_ch_names):
        raise IndexError('Channels {} not found'.format(
            set(upper_list_of_ch_names).difference(set(array(upper_montage_ch_names)[indices]))
        ))
    pos = montage.pos[indices, :2] if not azimuthal else azimuthal_equidistant_projection(montage.pos[indices, :3])
    return array(pos)


def validate_ch_names(list_of_ch_names, kind='standard_1005'):
    montage = read_montage(kind)
    upper_list_of_ch_names = [ch.upper() for ch in list_of_ch_names]
    upper_montage_ch_names = [ch.upper() for ch in montage.ch_names]
    if DEBUG or seems_to_come_from_neuromag(list_of_ch_names):
        bool_indices = [True for ch in upper_list_of_ch_names]
    else:
        bool_indices = [ch in upper_montage_ch_names for ch in upper_list_of_ch_names]
    return bool_indices


if __name__ == '__main__':
    # print(ch_names_to_2d_pos(['Cz', 'F8', 'F7', 'Cz']))
    print(ch_names_to_2d_pos(['Cz', 'Fp1']))
    print(validate_ch_names(['Cz', 'Cz0101']))


class WaitMessage(QtWidgets.QWidget):
    def __init__(self, text=''):
        super(WaitMessage, self).__init__()
        self.setWindowModality(QtCore.Qt.ApplicationModal)
        layout = QtWidgets.QHBoxLayout(self)
        layout.setAlignment(QtCore.Qt.AlignCenter)
        msg = QtWidgets.QLabel(text or 'Please wait ...')
        layout.addWidget(msg)
        self.setWindowFlags(QtCore.Qt.CustomizeWindowHint)
        self.resize(200, 100)

    def show_and_return(self):
        self.show()
        self.repaint()
        return self
