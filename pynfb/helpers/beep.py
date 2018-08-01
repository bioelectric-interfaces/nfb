from PyQt5.QtCore import QCoreApplication

class SingleBeep:
    def __init__(self):
        self._was = False

    def try_to_play(self):
        if not self._was:
            self._was = True
            try:
                QCoreApplication.instance().beep()
            except:
                print("Beep wasn't played")


if __name__ == '__main__':
    from PyQt5.QtWidgets import QApplication
    a = QApplication([])
    beep = SingleBeep()
    beep.try_to_play()
    from time import sleep
    sleep(3)
    beep.try_to_play()
    sleep(3)
    SingleBeep().try_to_play()
