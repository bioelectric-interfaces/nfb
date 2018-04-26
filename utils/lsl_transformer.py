from pynfb.inlets.lsl_inlet import LSLInlet
from pynfb.outlets.signals_outlet import SignalsOutlet
from time import sleep


class LSLTransformer:
    def __init__(self, inlet_name, outlet_name, outlet_channels_labels=None):
        # connect to inlet
        self.inlet = LSLInlet(inlet_name)
        # setup outlet
        outlet_fs = self.inlet.get_frequency()
        if outlet_channels_labels is None:
            outlet_channels_labels = self.inlet.get_channels_labels()
        # create outlet
        self.outlet = SignalsOutlet(outlet_channels_labels, outlet_fs, outlet_name)

    def update(self):
        inlet_chunk, input_timestamps = self.inlet.get_next_chunk()
        if inlet_chunk is not None:
            output_chunk = self.transform(inlet_chunk)
            self.outlet.push_chunk(output_chunk.T.tolist())

    def transform(self, x):
        return x**2


if __name__ == '__main__':
    decepticon = LSLTransformer('NVX136_Data', 'Tesseract_Data')
    while True:
        decepticon.update()
        sleep(0.05)