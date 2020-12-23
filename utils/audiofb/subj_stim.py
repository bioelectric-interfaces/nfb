from pynfb.helpers.simple_socket import SimpleServer
from time import sleep
from psychopy import visual, core


win = visual.Window([400,400])
message = visual.TextStim(win, text='Сообщение экспериментатору:\n запустите NFBLab')
message.autoDraw = True  # Automatically draw every frame
win.flip()

server = SimpleServer()

while 1:
        meta_str, obj = server.pull_message()
        if meta_str == 'msg':
            print('Dummy.. Set message to "{}"'.format(obj))
            message.text = obj
            win.flip()
        if meta_str == 'fb1':
            print('Dummy.. Run FB. Set message to "{}"'.format(obj))
            message.text = obj
            win.flip()
        if meta_str == 'spt':
            print('Dummy.. Set spatial filter to {}'.format(obj))
        if meta_str == 'bnd':
            print('Dummy.. Set band to {}'.format(obj))
        if meta_str == 'std':
            print('Dummy.. Set stats to {}'.format(obj))

        sleep(0.5)


