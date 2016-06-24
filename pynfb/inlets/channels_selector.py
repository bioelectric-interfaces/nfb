import numpy as np


class ChannelsSelector:
    def __init__(self, inlet, include=None, exclude=None, start_from_1=True):
        self.inlet = inlet

        # get channels indices to select
        if include:
            include = self.parse_channels_string(include)
            if isinstance(include, list):
                if isinstance(include[0], int):
                    include_indices = [j - int(start_from_1) for j in include]
                elif isinstance(include[0], str):
                    names = [n.upper() for n in self.inlet.get_channels_labels()]
                    include_indices = [names.index(r.upper()) for r in include]
                else:
                    raise TypeError('Reference list must contain int or str instances')
            else:
                raise TypeError('Reference list must be list or None')
        else:
            include_indices = list(range(self.inlet.get_n_channels()))

        # exclude channels

        if exclude:
            exclude = self.parse_channels_string(exclude)
            if isinstance(exclude, list):
                if isinstance(exclude[0], int):
                    exclude_indices = [j - int(start_from_1) for j in exclude]
                elif isinstance(exclude[0], str):
                    names = [n.upper() for n in self.inlet.get_channels_labels()]
                    exclude_indices = [names.index(r.upper()) for r in exclude]
                else:
                    raise TypeError('Exclude must contain int or str instances')
            else:
                raise TypeError('Exclude list must be list or None')
        else:
            exclude_indices = []

        # all indices
        self.indices = [j for j in include_indices if j not in exclude_indices]
        print(self.indices)

    def get_next_chunk(self):
        chunk = self.inlet.get_next_chunk()
        return chunk[:, self.indices] if (chunk is not None) else None

    def update_action(self):
        pass

    def save_info(self, file):
        return self.inlet.save_info(file)

    def get_frequency(self):
        return self.inlet.get_frequency()

    def get_n_channels(self):
        return len(self.indices)


    def get_channels_labels(self):
        return [self.inlet.get_channels_labels()[ind] for ind in self.indices]

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