import sys
from pynfb.inlets.FieldTrip import Client as FieldTrip_Client, numpyType
import time


class FieldTripBufferInlet:
    def __init__(self, host='localhost', port=1972):
        ftc = FieldTrip_Client()
        try:
            ftc.connect(host, port)  # might throw IOError
        except:
            print('Something went wrong while connecting to the Fieldtrip\n\
            buffer on localhost:1972. Did you start neuromag2ft on sinuhe?')
            raise SystemExit
        self.ftc = ftc
        self.last_repeated_sample = 0

    def get_next_chunk(self):
        H = self.ftc.getHeader()
        last_sample = H.nSamples - 1
        if last_sample == self.last_repeated_sample:  # no new data in FT buffer
            return None, None
        # If it is the first time then retrieve only one sample
        if self.last_repeated_sample == 0:
            retrieve_from = last_sample
        # Else retrieve all the new samples
        else:
            retrieve_from = self.last_repeated_sample + 1
        chunk = self.ftc.getData([retrieve_from, last_sample])
        self.last_repeated_sample = last_sample
        chunk = chunk.astype('float64')
        if len(chunk) == 0:
            return None, None
        else:
            return chunk, chunk[:, 0]*0  # chunk, timestamp mock

    def update_action(self):
        pass

    def save_info(self, file):
        with open(file, 'w', encoding="utf-8") as f:
            f.write(str(self.ftc.getHeader()))

    def info_as_xml(self):
        return str(self.ftc.getHeader())

    def get_frequency(self):
        return self.ftc.getHeader().fSample

    def get_n_channels(self):
        return self.ftc.getHeader().nChannels

    def get_channels_labels(self):
        labels = self.ftc.getHeader().labels
        if len(labels) == 0:
            labels = ['Ch{}'.format(k + 1) for k in range(self.get_n_channels())]
        return labels

    def disconnect(self):
        self.ftc.disconnect()


if __name__ == '__main__':
    inlet = FieldTripBufferInlet()
    while True:
        print(inlet.get_next_chunk())
