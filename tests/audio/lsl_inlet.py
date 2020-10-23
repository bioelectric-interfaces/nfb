# -*- coding: utf-8 -*-
import numpy as np
from pylsl import StreamInlet, resolve_byprop, resolve_bypred
from pylsl.pylsl import lib, StreamInfo, FOREVER, c_int, c_double, byref, handle_error
import xml.etree.ElementTree as ET
import time
import socket
fmt2string = ['undefined', 'float32', 'float64', 'str', 'int32', 'int16',
              'int8', 'int64']
LSL_STREAM_NAMES = ['AudioCaptureWin', 'NVX136_Data', 'example']
LSL_RESOLVE_TIMEOUT = 10


class FixedStreamInfo(StreamInfo):
    def as_xml(self):
        return lib.lsl_get_xml(self.obj).decode('utf-8', 'ignore') # add ignore

class FixedStreamInlet(StreamInlet):
    def info(self, timeout=FOREVER):
        errcode = c_int()
        result = lib.lsl_get_fullinfo(self.obj, c_double(timeout),
                                      byref(errcode))
        handle_error(errcode)
        return FixedStreamInfo(handle=result) # StreamInfo(handle=result)


class LSLInlet:
    def __init__(self, name=LSL_STREAM_NAMES[2], only_this_host=False):
        if not only_this_host:
            streams = resolve_byprop('name', name, timeout=LSL_RESOLVE_TIMEOUT)
        else:
            streams = resolve_bypred("name='{}' and hostname='{}'".format(name, socket.gethostname()))

        self.inlet = None
        self.dtype = 'float64'
        if len(streams) > 0:
            self.inlet = FixedStreamInlet(streams[0], max_buflen=2)
            # self.dtype = fmt2string[self.inlet.info().channel_format()]
            print(self.dtype)
            self.n_channels = self.inlet.info().channel_count()

    def get_next_chunk(self):
        # get next chunk
        chunk, timestamp = self.inlet.pull_chunk()
        # convert to numpy array
        chunk = np.array(chunk, dtype=self.dtype)
        # return first n_channels channels or None if empty chunk
        return (chunk, timestamp) if chunk.shape[0] > 0 else (None, None)

    def update_action(self):
        pass

    def save_info(self, file):
        with open(file, 'w', encoding="utf-8") as f:
            f.write(self.info_as_xml())

    def info_as_xml(self):
        xml = self.inlet.info().as_xml()
        return xml

    def get_frequency(self):
        return self.inlet.info().nominal_srate()

    def get_n_channels(self):
        return self.inlet.info().channel_count()

    def get_channels_labels(self):
        for t in range(3):
            time.sleep(0.5*(t+1))
            try:
                rt = ET.fromstring(self.info_as_xml())
                channels_tree = rt.find('desc').findall("channel") or rt.find('desc').find("channels").findall(
                    "channel")
                labels = [(ch.find('label') if ch.find('label') is not None else ch.find('name')).text
                          for ch in channels_tree]
                return labels
            except OSError:
                print('OSError during reading channels names', t+1)
        return ['channel'+str(n+1) for n in range(self.get_n_channels())]

    def disconnect(self):
        del self.inlet
        self.inlet = None
