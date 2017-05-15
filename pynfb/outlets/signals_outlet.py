from pylsl import StreamInfo, StreamOutlet
from numpy import repeat, reshape

class SignalsOutlet:
    def __init__(self, signals, fs):
        self.info = StreamInfo(name='NFBLab_data', type='', channel_count=len(signals), source_id='nfblab42',
                               nominal_srate=fs)
        self.info.desc().append_child_value("manufacturer", "BioSemi")
        channels = self.info.desc().append_child("channels")
        for c in signals:
            channels.append_child("channel").append_child_value("name", c)
        self.outlet = StreamOutlet(self.info)

    def push_sample(self, data):
        self.outlet.push_sample(data)

    def push_repeated_chunk(self, data, n=1):
        #chunk = repeat(data, n).reshape(-1, n).T.tolist()
        #self.outlet.push_chunk(chunk)
        for k in range(n):
            self.outlet.push_sample(data)




if __name__ == '__main__':
    outlet = SignalsOutlet(['alpha', 'beta'])
    print(outlet.info.as_xml())