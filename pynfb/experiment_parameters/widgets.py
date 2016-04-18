import pyqtgraph.parametertree.parameterTypes as pTypes
from pynfb.experiment_parameters.defaults import *
from pynfb.experiment_parameters.xml_io import *


def formatted_odict_to_params(odict):
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


class ScalableGroup(pTypes.GroupParameter):
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
        pTypes.GroupParameter.__init__(self, **opts)

    def addNew(self, type_, children=None):
        # print(self.vector_name, type_, children)
        # print(vectors_defaults[self.vector_name][type_])
        if self.vector_name == 'vPSequence':
            self.addChild(dict(name=type_ + str(len(self.childs) + 1),
                               type='str', value='', removable=True, renamable=False))
        else:
            if children is None:
                children = formatted_odict_to_params(vectors_defaults[self.vector_name][type_][0])
            self.addChild(dict(name=type_ + str(len(self.childs) + 1),
                               type='group', children=children, removable=True, renamable=False))

    def addNewStr(self, s):
        item = s
        item['removable'] = True
        item['name'] += str(len(self.childs) + 1)
        self.addChild(item)


def vector_odict_to_params(vdict):
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
    if not isinstance(odict, OrderedDict):
        return odict
    formatted_odict = OrderedDict()
    for key in defaults.keys():
        if key in odict.keys():
            if isinstance(defaults[key], OrderedDict):
                formatted_odict[key] = format_odict_by_defaults(odict[key], defaults[key])
            elif isinstance(defaults[key], list):
                #print(odict[key], defaults[key])
                formatted_odict[key] = [format_odict_by_defaults(item, defaults[key][0])
                                        for item in odict[key]]
            else:
                if isinstance(defaults[key], bool):
                    formatted_odict[key] = bool(odict[key])
                else:
                    formatted_odict[key] = odict[key]
        else:
            #print(key, defaults[key])
            formatted_odict[key] = defaults[key]
    return formatted_odict


if __name__ == '__main__':
    from pyqtgraph.Qt import QtCore, QtGui
    from pyqtgraph.parametertree import Parameter, ParameterTree
    import sys

    app = QtGui.QApplication([])
    odict = read_xml_to_dict('pynfb/experiment_parameters/settings/pilot.xml', True)
    #print(odict)
    params = formatted_odict_to_params(general_defaults)
    params = formatted_odict_to_params(format_odict_by_defaults(odict, general_defaults))
    #params += vector_odict_to_params(vectors_defaults)
    print(formatted_odict_to_params(format_odict_by_defaults(odict, vectors_defaults)))
    params += vector_odict_to_params(format_odict_by_defaults(odict, vectors_defaults))
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
    s = p.saveState()
    p.restoreState(s)

    ## Start Qt event loop unless running in interactive mode or using pyside.
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
