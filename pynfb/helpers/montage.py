from pynfb.widgets.helpers import ch_names_to_2d_pos


class Channel:
    def __init__(self, name, pos, type):
        '''
        :param name: str
        :param pos: tuple or numpy array of x, y, z coordinates (len = 3) or None
        :param type: str ['eeg', 'grad', 'mag', 'aux', 'other']
        '''
        self.name = name
        self.pos = pos
        self.type = type


class Montage:
    def __init__(self, ch_list):
        '''
        :param ch_list: list of Channel instances
        '''
        self.ch_list = ch_list

    @classmethod
    def create_from_names(cls, names, file_path=None):
        # put your code here ***********
        # you can use "ch_names_to_2d_pos()"
        # use file_path if not None
        ch_list = []
        return cls(ch_list)

    def pick_types(self, type):
        mask = []
        return mask

    def plot_topography(self, data, types, axes=None):
        assert len(types) == len(axes) or axes is None
        axes = axes
        return axes





if __name__ == '__main__':
    ch_info_from_lsl = ['Fp1', 'Fp2', 'Cz']
