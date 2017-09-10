import numpy as np
from pynfb.widgets.helpers import validate_ch_names



class ChannelsSelector:
    def __init__(self, inlet, include=None, exclude=None, start_from_1=True, subtractive_channel=None, dc=False):
        self.last_y = 0
        self.inlet = inlet
        names = [n.upper() for n in self.inlet.get_channels_labels()]
        names_ = [''.join([ch if ch.isalnum() else ' ' for ch in name]).split()[0] for name in names]
        self.channels_names = names
        print(names)
        names_isvalid = validate_ch_names(names)
        # if sum(names_isvalid) < len(names):
        #     print('WARNING: inlet channels {} are not in standard 1005 scheme'.format(
        #         [name for name, isvalid in zip(names, names_isvalid) if not isvalid]))


        # get channels indices to select
        if include:
            include = self.parse_channels_string(include)
            if isinstance(include, list):
                if isinstance(include[0], int):
                    include_indices = [j - int(start_from_1) for j in include]
                elif isinstance(include[0], str):
                    include_indices = [names.index(r.upper()) for r in include]
                else:
                    raise TypeError('Reference list must contain int or str instances')
            else:
                raise TypeError('Reference list must be list or None')
        else:
            include_indices = list(range(self.inlet.get_n_channels()))

        # channel to subtract
        if (subtractive_channel is not None) and (subtractive_channel != ''):
            if isinstance(subtractive_channel, int):
                self.sub_channel_index = subtractive_channel - int(start_from_1)
            elif isinstance(subtractive_channel, str):
                self.sub_channel_index = names.index(subtractive_channel.upper())
        else:
            self.sub_channel_index = None

        # exclude channels

        if exclude:
            exclude = self.parse_channels_string(exclude)
            if isinstance(exclude, list):
                if isinstance(exclude[0], int):
                    exclude_indices = [j - int(start_from_1) for j in exclude]
                elif isinstance(exclude[0], str):
                    print('Channels labels:', names)
                    print('Exclude:', [r.upper() for r in exclude])
                    exclude_indices = [names.index(r.upper()) for r in exclude if r.upper() in names]
                else:
                    raise TypeError('Exclude must contain int or str instances')
            else:
                raise TypeError('Exclude list must be list or None')
        else:
            exclude_indices = []

        # exclude not valid channels
        # exclude_indices += [j for j, isvalid in enumerate(names_isvalid) if not isvalid]

        # exclude subtractive channel
        if self.sub_channel_index is not None:
            if self.sub_channel_index not in exclude_indices:
                exclude_indices.append(self.sub_channel_index)

        # all indices
        exclude_indices = set(exclude_indices)
        self.indices = [j for j in include_indices if j not in exclude_indices]
        self.other_indices = [j for j in range(self.inlet.get_n_channels()) if j in exclude_indices]
        self.dc = dc


    def get_next_chunk(self):
        chunk = self.inlet.get_next_chunk()
        if chunk is not None:
            if self.dc:
                chunk = self.dc_blocker(chunk)
            if self.sub_channel_index is None:
                return chunk[:, self.indices], chunk[:, self.other_indices]
            else:
                return chunk[:, self.indices] - chunk[:, [self.sub_channel_index]], chunk[:, self.other_indices]
        else:
            return None, None


    def dc_blocker(self, x, r=0.99):
        # DC Blocker https://ccrma.stanford.edu/~jos/fp/DC_Blocker.html
        y = np.zeros_like(x)
        y[0] = self.last_y
        for n in range(1, x.shape[0]):
            y[n] = x[n] - x[n - 1] + r * y[n - 1]
        self.last_y = y[-1]
        return y

    def update_action(self):
        pass

    def save_info(self, file):
        try:
            return self.inlet.save_info(file)
        except UnicodeDecodeError:
            print("Warning: stream info wasn't saved, because user name id nonlatin")
            pass

    def info_as_xml(self):
        try:
            return self.inlet.info_as_xml()
        except UnicodeDecodeError:
            print("Warning: stream info wasn't saved, because user name id nonlatin")
            pass


    def get_frequency(self):
        return self.inlet.get_frequency()

    def get_n_channels(self):
        return len(self.indices)

    def get_n_channels_other(self):
        return len(self.other_indices)


    def get_channels_labels(self):
        return [self.channels_names[ind] for ind in self.indices]

    def disconnect(self):
        self.inlet.disconnect()

    @staticmethod
    def parse_channels_string(string):
        import re
        _list = [] if string == '' else re.split('; |, |\*|\n| |', string)
        if len(_list) > 0:
            try:
                _list = [int(e) for e in string]
            except (ValueError, TypeError):
                pass
        return _list