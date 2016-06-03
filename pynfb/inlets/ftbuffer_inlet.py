import sys
from pynfb.inlets.FieldTrip import Client as FieldTrip_Client, numpyType
import time

class FieldTripBufferInlet():
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
        if last_sample == self.last_repeated_sample: # no new data in FT buffer
            return
        # If it is the first time then retrieve only one sample
        if self.last_repeated_sample == 0:
            retrieve_from = last_sample
        # Else retrieve all the new samples
        else:
            retrieve_from = self.last_repeated_sample+1
        D = self.ftc.getData([retrieve_from, last_sample])
        self.last_repeated_sample = last_sample
        return D

    def update_action(self):
        pass

    def save_info(self, file):
        with open(file, 'w') as f:
            f.write(str(self.ftc.getHeader()))


    def get_frequency(self):
        return self.ftc.getHeader().fSample


    def get_n_channels(self):
        return self.ftc.getHeader().nChannels


    def disconnect(self):
        self.ftc.disconnect()


if __name__=='__main__':
    inlet = FieldTripBufferInlet()
    while True:
        print(inlet.get_next_chunk())


