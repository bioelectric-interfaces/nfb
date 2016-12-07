import h5py
import xml.etree.ElementTree as ET
import numpy as np
import seaborn
import pandas as pd
experiment_file = 'C:\\Users\\Nikolai\\Downloads\\pilot\\pilot_Emelyannikov29_1_11-09_20-14-28\\experiment_data.h5'

with h5py.File(experiment_file) as f:
    root = ET.fromstring(f['settings.xml'][:][0])
    sequence = [x.text for x in root.find('vPSequence')]
    types = dict([(x.find('sProtocolName').text, x.find('sFb_type').text) for x in root.find('vProtocols')])
    fb_keys = ['protocol' + str(k + 1) for k, protocol in enumerate(sequence) if types[protocol] == 'CircleFeedback']
    derived = [x.find('sSignalName').text for x in root.find('vSignals').findall('DerivedSignal')]
    composite = [x.find('sSignalName').text for x in root.find('vSignals').findall('CompositeSignal')]

    data_to_plot = pd.DataFrame(columns=['val', 'signal', 'protocol'])
    for k, fb_key in enumerate(fb_keys):
        for signal in derived:
            data_to_plot = data_to_plot.append(pd.DataFrame({'val': [f['{}/signals_stats/{}/mean'.format(fb_key, signal)].value],
                                'signal': signal,
                                'protocol': fb_key}
                                               ))

seaborn.barplot(data=data_to_plot, x='protocol', y='val', hue='signal')
seaborn.plt.show()

