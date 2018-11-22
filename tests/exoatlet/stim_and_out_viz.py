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

stim_rect = pygame.Rect((0,0), (screen_w,screen_h//2))
stim_color_list = [colors['r'], colors['y'], colors['g']]



images_dict = {}
for image_name in os.listdir('img'):
    name = image_name.split('.')[0]
    image = pygame.image.load(r'img\{}.png'.format(name))
    image = pygame.transform.scale(image, (3 * screen_h // 10, 3 * screen_h // 10))
    images_dict[name] = image
image_pos = (5*screen_w//10 - 3 * screen_h // 20, 6*screen_h//10)
images_name_list = ['stop', 'disable', 'go']


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
        set_stim_color(stim_color_list[state])
        if state>1: Beep(350, 100)

        # decoder
        out = chunk[0, -2]
        if out < 0.05:
            set_image('stop')
        elif out < 0.5:
            set_image('disable')
        else:
            set_image('go')

        pygame.display.update()
    time.sleep(0.1)