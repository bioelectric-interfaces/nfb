# -*- coding: utf-8 -*-
import numpy as np
from pylsl import StreamInlet, resolve_stream


LSL_STREAM_NAMES = ['AudioCaptureWin', 'NVX136_Data', 'example']


class LSLStream():
    def __init__(self, name='example', max_chunklen=8):
        streams = resolve_stream('name', name)
        self.inlet = StreamInlet(streams[0], max_buflen=1, max_chunklen=max_chunklen)

    def get_next_chunk(self):
        chunk, timestamp = self.inlet.pull_chunk()
        chunk = np.array(chunk)
        if chunk.shape[0] > 0:
            return chunk
        return None

    def update_action(self):
        pass
