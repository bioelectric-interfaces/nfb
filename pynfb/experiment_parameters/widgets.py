import pyqtgraph.parametertree.parameterTypes as pTypes
from pynfb.experiment_parameters.defaults import *
from pynfb.experiment_parameters.xml_io import *


class ScalableGroup(pTypes.GroupParameter):
    """ Class for scalable group in parameter tree
    """
    def __init__(self, **opts):
        self.vector_name = opts['name']
        opts['type'] = 'group'
        opts['addText'] = "add one"
        if self.vector_name == 'vSignals':
            opts['addList'] = ['DerivedSignal']
        if self.vector_name == 'vProtocols':
            opts['addList'] = ['FeedbackProtocol']
        if self.vector_name == 'vPSequence':
            opts['addList'] = ['s']
        super(ScalableGroup, self).__init__(**opts)

    def addNew(self, type_=None, children=None):
        if self.vector_name == 'vPSequence':
            self.addChild(dict(name=type_ + str(len(self.childs) + 1),
                               type='str', value='', removable=True, renamable=False))
        else:
            if children is None:
                children = formatted_odict_to_params(vectors_defaults[self.vector_name][type_][0])
            children[0]['value'] += str(len(self.childs) + 1)
            self.addChild(dict(name=type_ + str(len(self.childs) + 1),
                               type='group', children=children, removable=True, renamable=False))

    def addNewStr(self, s):
        item = s
        item['removable'] = True
        item['name'] += str(len(self.childs) + 1)
        self.addChild(item)


def formatted_odict_to_params(odict):
    """ Convert formatted ordered dict to params
    :param odict: ordered dict of parameters
    :return: list of params
    """
    params = []
    for k, v in odict.items():
        if isinstance(v, OrderedDict):
            params.append({'name': k, 'type': 'group', 'children': formatted_odict_to_params(v)})
        elif isinstance(v, list):
            children = []
            for i, ch in enumerate(v):
                if isinstance(ch, OrderedDict):
                    children.append({'name': k, 'type': 'group', 'children': formatted_odict_to_params(ch)})
                else:
                    children.append({'name': k, 'type': 'str', 'value': str(ch)})
            params += children
        else:
            params.append({'name': k, 'type': type(v).__name__, 'value': v})
    return params


def vector_formatted_odict_to_params(vdict):
    """ Convert formatted ordered dict of vector parameters to params
        :param odict: ordered dict of vector parameters
        :return: list of params
    """
    print(len(vdict), vdict)
    params = []
    for key in vdict.keys():
        group = ScalableGroup(name=key)
        items = formatted_odict_to_params(vdict[key])
        for item in items:
            if key == 'vPSequence':
                group.addNewStr(item)
            else:
                group.addNew(type_=item['name'], children=item['children'])
        params.append(group)
    return params


def format_odict_by_defaults(odict, defaults):
    """ Format ordered dict of params by defaults ordered dicts of parameters
    :param odict:  ordered dict
    :param defaults: defaults
    :return: ordered dict
    """
    if not isinstance(odict, OrderedDict):
        return odict
    formatted_odict = OrderedDict()
    for key in defaults.keys():
        if key in odict.keys():
            if isinstance(defaults[key], OrderedDict):
                formatted_odict[key] = format_odict_by_defaults(odict[key], defaults[key])
            elif isinstance(defaults[key], list):
                formatted_odict[key] = [format_odict_by_defaults(item, defaults[key][0])
                                        for item in odict[key]]
            else:
                if isinstance(defaults[key], bool):
                    formatted_odict[key] = bool(odict[key])
                else:
                    formatted_odict[key] = odict[key]
        else:
            formatted_odict[key] = defaults[key]
    return formatted_odict


def params_to_odict(params):
    odict = OrderedDict()
    for k, v in params.items():
        if isinstance(v, tuple):
            if v[0] is not None:
                odict[k] = v[0]
            else:
                odict[k] = params_to_odict(v[1])
        else:
            'not a tuple'
    for key in ['vSignals', 'vProtocols', 'vPSequence']:
        if key in odict:
            odict[key ] = [val for val in odict[key ].values()]
    return odict

if __name__ == '__main__':
    from pyqtgraph.Qt import QtCore, QtGui
    from pyqtgraph.parametertree import Parameter, ParameterTree
    import sys

    app = QtGui.QApplication([])

    # from defaults
    # params = formatted_odict_to_params(general_defaults)
    # params += vector_odict_to_params(vectors_defaults)

    # from file
    odict = read_xml_to_dict('pynfb/experiment_parameters/settings/pilot.xml', True)
    params = formatted_odict_to_params(format_odict_by_defaults(odict, general_defaults))
    params += vector_formatted_odict_to_params(format_odict_by_defaults(odict, vectors_defaults))

    # Create tree of Parameter objects
    p = Parameter.create(name='params', type='group', children=params)

    # Create two ParameterTree widgets, both accessing the same data
    t = ParameterTree()
    t.setParameters(p, showTop=False)
    t.setAutoFillBackground(False)
    t.setWindowTitle('pyqtgraph example: Parameter Tree')

    win = QtGui.QWidget()
    styleSheet = """
    QTreeView::item {
        border: 1px solid #d9d9d9;
        border-top-color: transparent;
        border-bottom-color: transparent;
        border-left-color: transparent;
        border-right-color: transparent;
        color: #000000
    }
    """
    win.setStyleSheet(styleSheet)
    layout = QtGui.QGridLayout()
    win.setLayout(layout)
    layout.addWidget(t, 1, 0, 1, 1)
    win.show()
    win.resize(800, 800)

    ## test save/restore
    # s = p.saveState()
    # p.restoreState(s)

    ## Start Qt event loop unless running in interactive mode or using pyside.
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
