from PyQt5 import QtCore, QtGui, QtWidgets
from pylsl import StreamInlet
from pylsl import resolve_bypred
from pynfb import STATIC_PATH


class ResolveButton(QtWidgets.QPushButton):
    def __init__(self):
        super(ResolveButton, self).__init__()
        self.text_str = 'Resolving '
        self.setStyleSheet ("text-align: left; padding-left: 10")
        self.setMinimumHeight(30)
        self.setMaximumWidth(90)
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update)
        self.timer.start(200)
        self.counter = 0

    def update(self):
        if not self.isEnabled():
            self.setText(self.text_str+' '.join(['.' for _ in range(self.counter)]))
        else:
            self.setText('Resolve again')
        self.counter += 1
        self.counter %= 4





class LSLResolveWaitWidget(QtWidgets.QWidget):
    def __init__(self, parent):
        super(LSLResolveWaitWidget, self).__init__(parent)

        # layout
        layout = QtWidgets.QVBoxLayout(self)

        # resolve thread
        self.resolve_thread = ResolveThread(self)
        self.resolve_thread.taskFinished.connect(self.onFinished)

        # refresh button
        self.refresh_btn = ResolveButton()
        #self.refresh_btn.setIcon(QtGui.QIcon(STATIC_PATH + '/imag/replay.png'))
        self.refresh_btn.clicked.connect(self.resolve)
        layout.addWidget(self.refresh_btn)

        # streams table
        self.streams_table = QtWidgets.QListWidget()
        layout.addWidget(self.streams_table)

        # resolve in init
        self.resolve()

    def fill_table(self, streams=None):
        if streams is None:
            streams = []

        def get_label(stream, n_channels):
            _info = stream
            print(_info.as_xml())
            import xml.etree.ElementTree as ET
            rt = ET.fromstring(_info.as_xml())
            channels_tree = rt.find('desc').findall("channel") or rt.find('desc').find("channels").findall("channel")
            #print(channels_tree[0].find('name'))
            labels = [(ch.find('label') if ch.find('label') is not None else ch.find('name')).text
                      for ch in channels_tree]
            return labels


        info = [(stream.name(), stream.type(), stream.channel_count(), stream.nominal_srate(),
                 get_label(stream, stream.channel_count())) for stream in streams]
        self.streams_table.clear()
        for inf in info:
            self.streams_table.addItem('{} ({}, {} channels, {} Hz)\n{}'.format(*inf))
        pass

    def onFinished(self):
        self.refresh_btn.setEnabled(True)
        self.fill_table(self.resolve_thread.streams)

    def resolve(self):
        self.refresh_btn.setEnabled(False)
        self.resolve_thread.start()

class ResolveThread(QtCore.QThread):
    taskFinished = QtCore.pyqtSignal()
    def __init__(self, widget):
        super(ResolveThread, self).__init__()
        self.widget = widget

    def run(self):
        streams = resolve_bypred("*", timeout=5)
        self.streams = [StreamInlet(stream).info() for stream in streams]
        self.taskFinished.emit()


if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = LSLResolveWaitWidget(None)
    window.show()
    sys.exit(app.exec_())
