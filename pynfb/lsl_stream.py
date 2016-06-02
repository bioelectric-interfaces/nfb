# -*- coding: utf-8 -*-
import numpy as np
from pylsl import StreamInlet, resolve_byprop
fmt2string = ['undefined', 'float32', 'float64', 'str', 'int32', 'int16',
              'int8', 'int64']


LSL_STREAM_NAMES = ['AudioCaptureWin', 'NVX136_Data', 'example']
LSL_RESOLVE_TIMEOUT = 10

class LSLStream():
    def __init__(self, name=LSL_STREAM_NAMES[2], max_chunklen=8):
        streams = resolve_byprop('name', name, timeout=LSL_RESOLVE_TIMEOUT)
        self.inlet = None
        self.dtype = 'float64'
        if len(streams) > 0:
            self.inlet = StreamInlet(streams[0], max_buflen=1, max_chunklen=max_chunklen)
            self.dtype = fmt2string[self.inlet.info().channel_format()]
            print(self.dtype)


    def get_next_chunk(self):
        # get next chunk
        chunk, timestamp = self.inlet.pull_chunk()
        # convert to numpy array
        chunk = np.array(chunk, dtype=self.dtype)
        # return first n_channels channels or None if empty chunk
        return chunk if chunk.shape[0] > 0 else None

    def update_action(self):
        pass

    def save_info(self, file):
        with open(file, 'w') as f:
            f.write(self.inlet.info().as_xml())
