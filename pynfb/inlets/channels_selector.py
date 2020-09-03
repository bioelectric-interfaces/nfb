import numpy as np
from pynfb.signal_processing.filters import ButterFilter, IdentityFilter
EVENTS_CHANNEL_NAME = 'EVENTS'


def interp_nans(y, empty_fill_val=0):
    nans = np.isnan(y)
    if sum(~nans) == 0:
        y = np.ones_like(y) * empty_fill_val
    else:
        y[nans] = np.interp(nans.nonzero()[0], (~nans).nonzero()[0], y[~nans])
    return y



class ChannelsSelector:
    def __init__(self, inlet, include=None, exclude=None, start_from_1=True, subtractive_channel=None, dc=False,
                 events_inlet=None, aux_inlets=None, aux_interpolate=False, prefilter_band=(None, None)):
        self.last_y = 0
        self.inlet = inlet
        self.events_inlet = events_inlet
        self.aux_inlets = aux_inlets
        self.aux_previous_chunks = []
        self.aux_interpolate = aux_interpolate

        # get names in uppercase format
        names = [n.upper() for n in self.inlet.get_channels_labels()]

        # cut after first non alphabetic numerical (e.g. 'Fp1-A1' -> 'Fp1')
        names = [''.join([ch if ch.isalnum() else ' ' for ch in name]).split()[0] for name in names]

        # append aux inlets
        if self.aux_inlets is not None:
            for aux_inlet in self.aux_inlets:
                names += aux_inlet.get_channels_labels()
                self.aux_previous_chunks.append(np.zeros_like((1, aux_inlet.n_channels)))

        # append events inlet
        if self.events_inlet is not None:
            names += [EVENTS_CHANNEL_NAME]
        self.channels_names = names
        print('Channels:', names)

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
            include_indices = list(range(len(names)))

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
        self.other_indices = [j for j in range(len(names)) if j in exclude_indices]
        self.dc = dc

        # pre-filtering settings
        if isinstance(prefilter_band, str):
            prefilter_band = [(float(s) if s != 'None' else None) for s in prefilter_band.split(' ')]
        if (prefilter_band[0] is None) and (prefilter_band[1] is None):
            self.prefilter = IdentityFilter()
        else:
            self.prefilter = ButterFilter(prefilter_band, self.inlet.get_frequency(),
                                          len(self.inlet.get_channels_labels()))


    def get_next_chunk(self):
        chunk, timestamp = self.inlet.get_next_chunk()
        if chunk is not None:
            if self.dc:
                chunk = self.dc_blocker(chunk)

            chunk = self.prefilter.apply(chunk)

            if self.events_inlet is not None:
                events, events_timestamp = self.events_inlet.get_next_chunk()
                aug_chunk = np.zeros((chunk.shape[0], 1))
                if events is not None:
                    aug_chunk[np.searchsorted(timestamp[:-1], events_timestamp)] = events
                chunk = np.hstack([chunk, aug_chunk])

            if self.aux_inlets is not None:
                for j_aux_inlet, aux_inlet in enumerate(self.aux_inlets):
                    aux_chunk_short, aux_timestamp = aux_inlet.get_next_chunk()
                    aux_chunk = np.zeros((chunk.shape[0], aux_inlet.n_channels)) * np.nan
                    if aux_chunk_short is not None:
                        for k in range(aux_inlet.n_channels):
                            aux_chunk[np.searchsorted(timestamp[:-1], aux_timestamp), k] = aux_chunk_short[:, k]
                            if self.aux_interpolate:
                                aux_chunk[:, k] = interp_nans(aux_chunk[:, k])
                        self.aux_previous_chunks[j_aux_inlet] = aux_chunk
                    else:
                        if self.aux_interpolate:
                            aux_chunk = np.ones((chunk.shape[0], aux_inlet.n_channels)) * \
                                        self.aux_previous_chunks[j_aux_inlet][-1]
                    chunk = np.hstack([chunk, aux_chunk])

            if self.sub_channel_index is None:
                return chunk[:, self.indices], chunk[:, self.other_indices], timestamp
            else:
                return chunk[:, self.indices] - chunk[:, [self.sub_channel_index]], chunk[:, self.other_indices], timestamp
        else:
            return None, None, None


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