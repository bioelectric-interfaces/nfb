from psychopy import gui, core, monitors
from psychopy.visual import Window
from psychopy.hardware import keyboard


mon = monitors.Monitor('eprime',
                 width=40,
                 distance=60,
                 autoLog=True)
mon.setSizePix((1280, 1024))

exp_info = {'participant': '99', 'session': ''}  # no default!
dlg = gui.DlgFromDict(exp_info)
if not dlg.OK:
    # Maybe add a nice print statement?
    print("User pressed 'Cancel'!")
    core.quit()
print(f"Started the experiment for participant {exp_info['participant']} in session {exp_info['session']}!")

win = Window(fullscr=False,
             monitor=mon,
             units='deg',
             screen=1)

clock = core.Clock()
kb = keyboard.Keyboard()
core.wait(2)
keys = kb.getKeys()
print(keys)

