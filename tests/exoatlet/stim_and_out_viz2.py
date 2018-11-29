import pygame, sys, time, os
from pynfb.inlets.lsl_inlet import LSLInlet
from winsound import Beep

colors = dict(r=(193, 53, 37), y=(255, 187, 67), g=(109, 160, 111))

inlet = LSLInlet('Exp_Data')

pygame.init()
infoObject = pygame.display.Info()
screen_w = 800
screen_h = 1000
screen=pygame.display.set_mode((screen_w,screen_h))
clock = pygame.time.Clock()

stim_rect = pygame.Rect((0,screen_h//2), (screen_w,screen_h//2))
stim_color_list = [colors['r'], colors['y'], colors['g']]



images_dict = {}
for image_name in os.listdir('img'):
    name = image_name.split('.')[0]
    image = pygame.image.load(r'img\{}.png'.format(name))
    image = pygame.transform.scale(image, (3 * screen_h // 10, 3 * screen_h // 10))
    images_dict[name] = image
image_pos = (5*screen_w//10 - 3 * screen_h // 20, 1*screen_h//10)
images_name_list = ['stop', 'disable', 'go']


# perf
result = None
prev_state = 0
is_disable = False

def set_stim_color(color):
    pygame.draw.rect(screen, color, stim_rect)

def set_image(image_name):
    screen.blit(images_dict[image_name], image_pos)

while True:
    for event in pygame.event.get():
        if event.type==pygame.QUIT:
             pygame.quit()
             sys.exit()
    chunk, _ = inlet.get_next_chunk()
    if chunk is not None:
        screen.fill((255, 255, 255))
        print(len(chunk))

        # state
        state = int(chunk[0, -1])



        if state != 1:
            prev_state = state
            if is_disable:
                result = 0
                is_disable = False
                if state == 2:
                    Beep(500, 70), Beep(1000, 100)

        else:
            if not is_disable and prev_state==2:
                Beep(1000, 70), Beep(500, 100)

            is_disable = True

        # decoder
        out = chunk[0, -3]

        #set_stim_color(stim_color_list[state])
        if state == 0:
            set_image('stop')
        if state == 2:
            set_image('go')


        if out < 0.05:
            #set_image('stop')
            set_stim_color(colors['r'])
        elif out < 0.5:
            if state!=1:
                #set_image('disable')
                set_stim_color(colors['y'])
        else:
            # set_image('go')
            set_stim_color(colors['g'])
            if state != 1:
                result = 1

        if out > 0.5 and state == 0:
            Beep(300, 100)

        if state == 1:
            if (prev_state == 0 and result == 0) or (prev_state == 2 and result == 1):
                set_image('like')
            else:
                set_image('dislike')

        pygame.display.update()

    time.sleep(0.1)