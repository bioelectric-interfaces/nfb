import expyriment

exp = expyriment.design.Experiment(name="First Experiment")
expyriment.io.Screen(window_mode=True)
expyriment.control.initialize(exp)

stim = expyriment.stimuli.TextLine(text="Hello World")
stim.preload()
expyriment.control.defaults.window_mode = True
expyriment.control.start(skip_ready_screen=True, subject_id=1)

stim.present()
exp.clock.wait(1000)

expyriment.control.end()