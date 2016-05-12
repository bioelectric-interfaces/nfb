# -*- coding: utf-8 -*-
import numpy as np
from pylsl import StreamInlet, resolve_stream


LSL_STREAM_NAMES = ['AudioCaptureWin', 'NVX136_Data', 'example']


class LSLStream():
    def __init__(self, name=LSL_STREAM_NAMES[2], max_chunklen=8, n_channels=32):
        self.n_channels = n_channels
        streams = resolve_stream('name', name)
        self.inlet = StreamInlet(streams[0], max_buflen=1, max_chunklen=max_chunklen)

    def get_next_chunk(self):
        # get next chunk
        chunk, timestamp = self.inlet.pull_chunk()
        # convert to numpy array
        chunk = np.array(chunk)
        # return first n_channels channels or None if empty chunk
        return chunk[:, :self.n_channels] if chunk.shape[0] > 0 else None

    def update_action(self):
        pass
