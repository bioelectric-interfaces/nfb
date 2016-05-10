from pynfb.signals import DerivedSignal
from pynfb.lsl_stream import LSLStream, LSL_STREAM_NAMES
from pynfb.protocols import BaselineProtocol, FeedbackProtocol
from PyQt4 import QtGui, QtCore
import sys
from pynfb.windows import  MainWindow
import numpy as np

class Experiment():
    def __init__(self, app, params):
        # inlet frequency
        self.freq = 500
        # signals
        int_or_none = lambda string: int(string) if len(string)>0 else None
        if 'vSignals' in params:
            self.signals = [DerivedSignal(bandpass_high=int_or_none(signal['fBandpassHighHz']),
                                          bandpass_low=int_or_none(signal['fBandpassLowHz']),
                                          name=signal['sSignalName'])
                            for signal in params['vSignals']]
        else:
            self.signals = [DerivedSignal(bandpass_low=8, bandpass_high=12), DerivedSignal(bandpass_low=40, bandpass_high=50)]
        self.current_samples = np.zeros_like(self.signals)
        # protocols
        if 'vProtocols' in params:
            self.protocols_list = []
            for protocol in params['vProtocols']:
                if protocol['sFb_type']=='':
                    self.protocols_list.append(BaselineProtocol(self.signals,
                                                                duration=int_or_none(protocol['fDuration']),
                                                                name=protocol['sProtocolName']))
                else:
                    self.protocols_list.append(FeedbackProtocol(self.signals,
                                                                duration=int_or_none(protocol['fDuration']),
                                                                name=protocol['sProtocolName']))
        else:
            pass

        if 'vPSequence' in params:
            names = [protocol.name for protocol in self.protocols_list]
            self.protocols = []
            for name in params['vPSequence']:
                self.protocols.append(self.protocols_list[names.index(name)])
            print(self.protocols)
        #self.protocols = [BaselineProtocol(self.signals, duration=10, base_signal_id=0),
        #                  FeedbackProtocol(self.signals, duration=10),
        #                  BaselineProtocol(self.signals, duration=10)]

        self.current_protocol_ind = 0
        self.main = MainWindow(signals=self.signals, parent=None, current_protocol=self.protocols[self.current_protocol_ind], n_signals=len(self.signals))
        self.subject = self.main.subject_window
        self.stream = LSLStream()
        # timer
        main_timer = QtCore.QTimer(app)
        main_timer.timeout.connect(self.update)
        main_timer.start(1000 * 1. / self.freq)
        # samples counter for protocol sequence
        self.samples_counter = 0
        self.current_protocol_n_samples = self.freq * self.protocols[self.current_protocol_ind].duration
        pass

    def update(self):
        chunk = self.stream.get_next_chunk()
        if chunk is not None:
            self.samples_counter += chunk.shape[0]
            for i, signal in enumerate(self.signals):
                signal.update(chunk)
                self.current_samples[i] = signal.current_sample
            self.main.redraw_signals(self.current_samples, chunk)
            self.subject.update_protocol_state(self.current_samples, chunk_size=chunk.shape[0])
            if self.samples_counter >= self.current_protocol_n_samples:
                self.next_protocol()

    def next_protocol(self):
        print('protocol:', self.current_protocol_ind, 'samples:', self.samples_counter)

        self.samples_counter = 0
        if self.current_protocol_ind < len(self.protocols)-1:
            self.protocols[self.current_protocol_ind].close_protocol()
            if self.protocols[self.current_protocol_ind].update_statistics_in_the_end:
                self.main.signals_buffer *= 0
            self.current_protocol_ind += 1
            self.current_protocol_n_samples = self.freq * self.protocols[self.current_protocol_ind].duration
            self.subject.change_protocol(self.protocols[self.current_protocol_ind])
        else:
            print('finish')
            self.current_protocol_n_samples = np.inf
            self.subject.close()

    def run(self):
        app = QtGui.QApplication(sys.argv)


if __name__=='__main__':
    app = QtGui.QApplication(sys.argv)
    experiment = Experiment(app)
    sys.exit(app.exec_())
    #experiment.run()
