from pynfb.io.xmltodict import parse, unparse
from collections import OrderedDict
from pynfb.io.defaults import *


def format_odict_by_defaults(odict, defaults):
    """ Format ordered dict of params by defaults ordered dicts of parameters
    :param odict:  ordered dict
    :param defaults: defaults
    :return: ordered dict
    """
    formatted_odict = OrderedDict()
    for key in defaults.keys():
        if key in odict.keys():
            if key in ['DerivedSignal', 'FeedbackProtocol']:
                formatted_odict[key] = [format_odict_by_defaults(item, defaults[key][0])
                                        for item in (odict[key] if isinstance(odict[key], list) else [odict[key]])]
            elif isinstance(defaults[key], OrderedDict):
                formatted_odict[key] = format_odict_by_defaults(odict[key], defaults[key])
            else:
                formatted_odict[key] = odict[key]
        else:
            formatted_odict[key] = defaults[key]
    return formatted_odict


def xml_file_to_odict(filename):
    """ Read xml to ordered dict
    :param filename: path to file
    :param skip_root: if True skip root
    :return: OrderedDict instance
    """
    # postprocessor convert to int if possible
    def postprocessor(path, key, value):
        if value is None:
            value = ''
        try:
            value = int(value)
        except (ValueError, TypeError):
            pass
        return key, value
    # read and parse
    with open(filename, 'r') as f:
        d = parse(f.read(), postprocessor=postprocessor)

    d = list(d.values())[0]

    return d


def xml_file_to_params(filename):
    d = xml_file_to_odict(filename)
    d = format_odict_by_defaults(d, vectors_defaults)
    d['vSignals'] = d['vSignals']['DerivedSignal']
    d['vProtocols'] = d['vProtocols']['FeedbackProtocol']
    d['vPSequence'] = d['vPSequence']['s']
    #print(d)
    return d


def params_to_xml_file(params, filename):
    print(params)
    odict = params
    odict['vSignals'] = OrderedDict([('DerivedSignal', params['vSignals'])])
    odict['vProtocols'] = OrderedDict([('FeedbackProtocol', params['vProtocols'])])
    odict['vPSequence'] = OrderedDict([('s', params['vPSequence'])])
    xml_odict = OrderedDict([('NeurofeedbackSignalSpecs', odict.copy())])
    with open(filename, 'w') as f:
        f.write(unparse(xml_odict, pretty=True))
    pass


if __name__ == '__main__':
    odict = xml_file_to_params('settings/pilot.xml')
    params_to_xml_file(odict, 'settings/pilot_rewrite.xml')