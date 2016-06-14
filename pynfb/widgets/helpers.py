try:
    from mne.channels import read_montage
    from mne import pick_channels
except ImportError:
    pass
from numpy import array


def ch_names_to_2d_pos(list_of_ch_names):
    montage = read_montage('standard_1005')
    indices = pick_channels(montage.ch_names, list_of_ch_names)
    if len(list(indices)) < len(list_of_ch_names):
        raise IndexError('Channels {} not found'.format(
            set(list_of_ch_names).difference(set(array(montage.ch_names)[indices]))
        ))
    pos = montage.pos[indices, :2]
    return pos

if __name__ == '__main__':
    print(ch_names_to_2d_pos(['Cz']))
    print(ch_names_to_2d_pos(['Cz1, Fp1']))