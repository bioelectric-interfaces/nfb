from . import read_spatial_filter
from .xmltodict import parse, unparse
from collections import OrderedDict
from .defaults import *
from numpy import array
import xml.etree.ElementTree as ET
from pynfb.signals import DerivedSignal



def format_odict_by_defaults(odict, defaults):
    """ Format ordered dict of params by defaults ordered dicts of parameters
    :param odict:  ordered dict
    :param defaults: defaults
    :return: ordered dict
    """
    formatted_odict = OrderedDict()
    for key in defaults.keys():
        if key in odict.keys():
            if key in ['DerivedSignal', 'FeedbackProtocol', 'CompositeSignal', 'PGroup']:
                if odict[key] == '':
                    formatted_odict[key] = []
                else:
                    formatted_odict[key] = [format_odict_by_defaults(item, defaults[key][0])
                                        for item in (odict[key] if isinstance(odict[key], list) else [odict[key]])]
            elif isinstance(defaults[key], OrderedDict):
                formatted_odict[key] = format_odict_by_defaults(odict[key], defaults[key])
            else:
                formatted_odict[key] = odict[key]
        else:
            formatted_odict[key] = defaults[key]
    return formatted_odict


def xml_file_to_odict(filename_or_str):
    """ Read xml to ordered dict
    :param filename_or_str: path to file or xml str
    :param skip_root: if True skip root
    :return: OrderedDict instance
    """
    # postprocessor convert to int if possible
    def postprocessor(path, key, value):
        if value is None:
            value = ''
        try:
            try:
                value = int(value)
            except ValueError:
                value = float(value)
        except (ValueError, TypeError):
            pass
        return key, value
    # read and parse
    if '<NeurofeedbackSignalSpecs>' not in filename_or_str:
        try:
            with open(filename_or_str, 'r', encoding="utf-8") as f:
                d = parse(f.read(), postprocessor=postprocessor)
        except UnicodeDecodeError:
            with open(filename_or_str, 'r', encoding="cp1251") as f:
                d = parse(f.read(), postprocessor=postprocessor)
    else:
        d = parse(filename_or_str, postprocessor=postprocessor)

    d = list(d.values())[0]

    return d


def xml_file_to_params(filename=None):
    d = vectors_defaults if filename is None else xml_file_to_odict(filename)
    d = format_odict_by_defaults(d, vectors_defaults)
    #d['vSignals'] = d['vSignals']['DerivedSignal']
    d['vProtocols'] = d['vProtocols']['FeedbackProtocol']
    protocols_sequence = d['vPSequence']['s']
    d['vPSequence'] = [protocols_sequence] if isinstance(protocols_sequence, str) else protocols_sequence
    #print(d)
    return d


def params_to_xml_file(params, filename):
    with open(filename, 'w', encoding="utf-8") as f:
        f.write(params_to_xml(params))

def params_to_xml(params):
    odict = params.copy()
    # odict['vSignals'] = OrderedDict([('DerivedSignal', params['vSignals'])])
    odict['vProtocols'] = OrderedDict([('FeedbackProtocol', params['vProtocols'])])
    odict['vPSequence'] = OrderedDict([('s', params['vPSequence'])])
    xml_odict = OrderedDict([('NeurofeedbackSignalSpecs', odict.copy())])
    def preprocessor(key, val):
        if key in ['DerivedSignal', 'FeedbackProtocol', 'CompositeSignal', 'PGroup'] and val == []:
            val = ''
        return key, val
    xml = unparse(xml_odict, pretty=True, preprocessor=preprocessor)

    return xml


def save_signal(signal, filename):
    default = vectors_defaults['vSignals']['DerivedSignal'][0].copy()
    default['sSignalName'] = signal.name
    default['fBandpassLowHz'] = signal.bandpass[0]
    default['fBandpassHighHz'] = signal.bandpass[1]
    default['SpatialFilterMatrix'] = signal.spatial_matrix
    default['bDisableSpectrumEvaluation'] = int(signal.disable_spectrum_evaluation)
    default['fFFTWindowSize'] = signal.n_samples
    default['fSmoothingFactor'] = signal.smoothing_factor
    signal_dict = OrderedDict([('DerivedSignal', default)])
    with open(filename, 'w', encoding="utf-8") as f:
        f.write(unparse(signal_dict, pretty=True))


def load_signal(filename, channels_labels):
    signal = xml_file_to_odict(filename)
    default = vectors_defaults['vSignals']['DerivedSignal'][0].copy()
    for key, value in default.items():
        default[key] = signal.get(key, value)
    signal = default

    if isinstance(signal['SpatialFilterMatrix'], str):
        if signal['SpatialFilterMatrix'] == '':
            spatial_filter = None
        else:
            spatial_filter = read_spatial_filter(signal['SpatialFilterMatrix'],
                                                 channels_labels)
    elif isinstance(signal['SpatialFilterMatrix'], list):
        spatial_filter = array(signal['SpatialFilterMatrix']).astype(float)
    else:
        raise TypeError ('\'SpatialFilterMatrix\' must be string or list (vector)')

    s = DerivedSignal(bandpass_high=signal['fBandpassHighHz'],
                      bandpass_low=signal['fBandpassLowHz'],
                      name=signal['sSignalName'],
                      n_channels=len(channels_labels),
                      spatial_filter=spatial_filter,
                      disable_spectrum_evaluation=signal['bDisableSpectrumEvaluation'],
                      n_samples=signal['fFFTWindowSize'],
                      smoothing_factor=signal['fSmoothingFactor'])
    return s

def get_lsl_info_from_xml(xml_str_or_file):
    try:
        tree = ET.parse(xml_str_or_file)
        root = tree.getroot()
    except (FileNotFoundError, OSError):
        root = ET.fromstring(xml_str_or_file)
    info = {}
    channels = [k.find('label').text for k in root.find('desc').find('channels').findall('channel')]
    fs = int(root.find('nominal_srate').text)
    return channels, fs


if __name__ == '__main__':
    #odict = xml_file_to_params('settings/pilot.xml')
    #params_to_xml_file(odict, 'settings/pilot_rewrite.xml')
    from pynfb.serializers.hdf5 import load_xml_str_from_hdf5_dataset
    xml_str = load_xml_str_from_hdf5_dataset('../results/experiment_03-07_13-23-46/experiment_data.h5',
                                             'stream_info.xml')
    print(get_lsl_info_from_xml(xml_str))

    fname = '../test_groups'
    params = xml_file_to_params(fname)
    print(params)

    params_to_xml_file(params, 'settings_test.xml')