try:
    from mne.channels import read_montage
    from mne import pick_channels
except ImportError:
    pass
from numpy import array


def ch_names_to_2d_pos(list_of_ch_names, kind='standard_1005'):
    montage = read_montage(kind)
    upper_list_of_ch_names = [ch.upper() for ch in list_of_ch_names]
    upper_montage_ch_names = [ch.upper() for ch in montage.ch_names]
    indices = [upper_montage_ch_names.index(ch) for ch in upper_list_of_ch_names if ch in upper_montage_ch_names]
    if len(list(indices)) < len(list_of_ch_names):
        raise IndexError('Channels {} not found'.format(
           set(upper_list_of_ch_names).difference(set(array(upper_montage_ch_names)[indices]))
        ))
    pos = montage.pos[indices, :2]
    return array(pos)

if __name__ == '__main__':
    #print(ch_names_to_2d_pos(['Cz', 'F8', 'F7', 'Cz']))
    print(ch_names_to_2d_pos(['Cz', 'Fp1']))