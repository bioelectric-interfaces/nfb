from psychopy.gui import DlgFromDict
from psychopy.visual import Window, TextStim
from psychopy.core import Clock, quit, wait
from psychopy.event import Mouse
from psychopy.hardware.keyboard import Keyboard
from psychopy.monitors import Monitor

### DIALOG BOX ROUTINE ###
exp_info = {'participant_nr': 99, 'age': ''}
dlg = DlgFromDict(exp_info)

# If pressed Cancel, abort!
if not dlg.OK:
    quit()
else:
    # Quit when either the participant nr or age is not filled in
    if not exp_info['participant_nr'] or not exp_info['age']:
        quit()

    # Also quit in case of invalid participant nr or age
    if exp_info['participant_nr'] > 99 or int(exp_info['age']) < 18:
        quit()
    else:  # let's star the experiment!
        print(f"Started experiment for participant {exp_info['participant_nr']} "
              f"with age {exp_info['age']}.")

# init the monitor
mon = Monitor('eprime',
                 width=40,
                 distance=60,
                 autoLog=True)
mon.setSizePix((1280, 1024))

# Initialize a fullscreen window with my monitor (HD format) size
# and my monitor specification called "samsung" from the monitor center
win = Window(fullscr=False, monitor=mon)

# Also initialize a mouse, for later
# We'll set it to invisible for now
mouse = Mouse(visible=False)

# Initialize a (global) clock
clock = Clock()

# Initialize Keyboard
kb = Keyboard()

### START BODY OF EXPERIMENT ###
#
welcome_txt_stim = TextStim(win, text="Welcome to this experiment!")
welcome_txt_stim.draw()
win.flip()
#
### END BODY OF EXPERIMENT ###

# Finish experiment by closing window and quitting
win.close()
quit()