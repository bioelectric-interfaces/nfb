import sys
import os
import pygame
import numpy as np
import random
import time
from pynfb.inlets.lsl_inlet import LSLInlet

STATIC_PATH = os.path.realpath(os.path.dirname(os.path.realpath(__file__)) + '/static')

white = (255, 255, 255)

pygame.init()
display_width, display_hight = 1440, 900
gameDisplay = pygame.display.set_mode((display_width, display_hight), (pygame.FULLSCREEN | pygame.HWSURFACE))
pygame.display.set_caption("Space Race")
random.seed(time.time())
clock = pygame.time.Clock()

gameDisplay.fill((0, 0, 0))
pygame.display.update()

random.seed(time.time() * 1000)

print(STATIC_PATH)

# Essentials

def load_image(name):
    """ Load image and return image object"""
    fullname = os.path.join(STATIC_PATH, 'images', name)
    image = pygame.image.load(fullname)
    return image


def load_sound(name):
    """ Load sound and return sound object"""
    fullname = os.path.join(STATIC_PATH, 'sounds', name)
    sound = pygame.mixer.Sound(fullname)
    return sound


def text_object(text, font):
    """ Returns PyGame text object and it's size """
    TextSurface = font.render(text, True, white)
    return TextSurface, TextSurface.get_rect()


def message_display(text):
    """ Makes a text message in the center of a screen """
    largeText = pygame.font.Font(r'C:\Users\Nikolai\PycharmProjects\nfb\tests\bci_test\static\9921.otf', 90)
    TextSurf, TextRect = text_object(text, largeText)
    TextRect.center = ((display_width / 2), (display_hight / 2))
    gameDisplay.blit(TextSurf, TextRect)


def demons_dodged(count, distance):
    font = pygame.font.Font(r'C:\Users\Nikolai\PycharmProjects\nfb\tests\bci_test\static\9921.otf', 30)
    text = font.render('Scores: {}'.format(str(count)), True, white)
    gameDisplay.blit(text, (5, 0))


# Characters


def spacecraft(img, x, y):
    gameDisplay.blit(img, (x, y))

def ring(coinX, coinY, coin_size, img):
    img = pygame.transform.scale(img, (coin_size, coin_size))
    gameDisplay.blit(img, (coinX, coinY))


# Passive Loops


def crash(img1, img2, bg, c, x, y):
    for i in range(3):
        message_display('GAME OVER!')
        pygame.time.wait(200)
        gameDisplay.blit(img1, (x, y))
        pygame.display.update()

        pygame.time.wait(200)
        gameDisplay.blit(bg, (0, 0 + c))
        gameDisplay.blit(img2, (x, y))
        message_display('GAME OVER!')
        pygame.display.update()

        gameDisplay.blit(bg, (0, 0 + c))
    game_loop()

# Active Loops

def menu_loop():
    intro = True
    title = True

    backgroundImgHight = backgroundImg.get_rect().size[1]
    print(backgroundImgHight)
    planet = load_image('p1.png')

    menu = [new_game, load_game, settings, credits]
    menu_id = 0

    planet_X = -150
    planet_Y = display_hight / 4
    planet_movementX = 0
    planet_movementY = 0

    planet_hight = 200
    planet_width = 200
    planet_hight_distortion = 0
    planet_width_distortion = 0

    parallax = 0

    planet = pygame.transform.scale(planet, (planet_hight + 300, planet_width + 300))

    while intro:

        clock.tick(60)

        if parallax == backgroundImgHight:
            parallax = 0

        gameDisplay.blit(backgroundImg, (0, (0 - backgroundImgHight) + parallax))
        gameDisplay.blit(backgroundImg, (0, 0 + parallax))

        gameDisplay.blit(planet, (planet_X + planet_movementX, planet_Y + planet_movementY))

        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                pygame.mixer.Sound.play(change)
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    quit()
                if event.key == pygame.K_DOWN:
                    if menu_id < 4:
                        menu_id += 1
                    if menu_id == 4:
                        menu_id = 0
                if event.key == pygame.K_UP:
                    if menu_id == -1:
                        menu_id = 3
                    if menu_id > -1:
                        menu_id -= 1
                if event.key == pygame.K_RETURN and menu_id == 0:
                    game_loop()

        if title:
            gameDisplay.blit(title_1, (0, 0))
            title = False
        else:
            gameDisplay.blit(title_2, (0, 0))
            title = True

        parallax += 1

        planet_movementX += 1
        planet_movementY = planet_movementX * 0.25

        planet_width_distortion += 1
        planet_hight_distortion += 1

        gameDisplay.blit(menu[menu_id], (0, 0))
        pygame.display.update()


def game_loop():
    backgroundImgHight = backgroundImg.get_rect().size[1]

    x = display_width * 0.45  # OX spacecraft start position
    y = display_hight * 0.8  # OY spacecraft start position

    x_change = 0

    coin = coins[0]
    coin_size = 65
    coin_startX = random.randrange(500, display_width-500)
    coin_startY = -200
    coin_speed = 2

    planet = planets[0]
    planet_size = 500
    planet_startX = random.randrange(-300, display_width-300)
    planet_startY = -600
    planet_speed = 1

    gameExit = False
    speed_up = False

    score = 0
    distance = 0
    counter = 0
    coin_state = 1
    parallax = 0
    sps = 0
    boost = 0
    delayB = 0
    delayM = 0
    delayP = 0

    pygame.mixer.Sound.play(soundtrack, loops=-1)

    lsl = LSLInlet(name='NFBLab_data')
    state = 0


    while not gameExit:
        chunk = lsl.get_next_chunk()
        if chunk is not None:
            state = chunk[-1, 0]

        if state == 1:
            x_change = -4
        elif state == 2:
            x_change = 4

        clock.tick(60)  # frames per second
        parallax += 1
        parallax += boost
        if parallax >= backgroundImgHight:
            parallax = 0

        for event in pygame.event.get():  # list of events per t

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    quit()
                if event.key == pygame.K_LEFT:
                    x_change = -20
                if event.key == pygame.K_RIGHT:
                    x_change = 20

            if event.type == pygame.KEYUP:
                if event.key == pygame.K_LEFT or event.key == pygame.K_RIGHT:
                    x_change = 0


        x += x_change

        gameDisplay.blit(backgroundImg, (0, 0 - backgroundImgHight + parallax))
        gameDisplay.blit(backgroundImg, (0, 0 + parallax))

        counter += 1

        if counter > 10:
            counter = 0

        # ring(planet_startX, planet_startY, planet_size, planet)
        ring(coin_startX, coin_startY, coin_size, coin)
        coin_startY += coin_speed+boost
        # planet_startY += planet_speed+boost
        distance += 1 + boost
        demons_dodged(score, distance)

        if counter > 5:
            spacecraft(spacecraftImg, x, y)
        else:
            spacecraft(spacecraft2Img, x, y)

        if x > display_width - spacecraft_width:
            #pygame.mixer.Sound.stop(soundtrack)
            #pygame.mixer.Sound.play(returns)
            #crash(explosion_1, explosion_2, backgroundImg, parallax, x, y)
            x = display_width - spacecraft_width

        if x < 0:
            x = 0

        if coin_startY > display_hight:
            coin_startY = -coin_size
            coin_startX = random.randrange(50, display_width - 50)

        if planet_startY > display_hight:
            planet = planets[random.randrange(len(planets))]
            planet_startY = -planet_size
            planet_startX = random.randrange(500, display_width - 500)

        if coin_state < 10:
            coin = coins[0]
        elif 10 <= coin_state < 20:
            coin = coins[1]
        elif 20 <= coin_state < 30:
            coin = coins[2]
        elif 30 <= coin_state < 40:
            coin = coins[3]

        coin_state += 1
        if coin_state >= 40:
            coin_state = 1

        if y <= coin_startY + coin_size:
            if x > coin_startX and x < coin_startX + coin_size or x + spacecraft_width > coin_startX and x + spacecraft_width < coin_startX + coin_size or x < coin_startX and x + spacecraft_width > coin_startX + coin_size:
                pygame.mixer.Sound.play(beep)
                coin_startY = -coin_size
                coin_startX = random.randrange(500, display_width - 500)
                # crash(explosion_1, explosion_2, backgroundImg, parallax, x, y)
                score += 1
                speed_up = True

        if speed_up == True:
            sps += 0
            # delayB = 30
            speed_up = False

        if sps > 0 and delayP == 0:
            boost += 1
            sps -= 1
            delayP = 0
        elif sps > 0:
            delayP -= 1

        if sps == 0 and boost > 0 and delayM == 0:
            boost -= 1
            delayM = 10
        if sps == 0 and boost > 0:
            delayM -= 1

        if boost > 30:
            boost = 30

        pygame.display.update()  # .flip()


# loading sounds

soundtrack = load_sound('soundtrack.wav')
beep = load_sound('Beep17.wav')
change = load_sound('change.wav')
returns = load_sound('return.wav')

# loading spacecraft image

spacecraftImg = load_image('spacecraft.png')
spacecraft2Img = load_image('spacecraft_2.png')
spacecraft_width = 150
spacecraftImg = pygame.transform.scale(spacecraftImg, (spacecraft_width, spacecraft_width))
spacecraft2Img = pygame.transform.scale(spacecraft2Img, (spacecraft_width, spacecraft_width))

# loading background image

backgroundImg = load_image('background.png')
# backgroundImg = pygame.transform.rotate(backgroundImg)  # it was 90 angle

# loading explosion image

explosion_1 = load_image('explosion_1.tiff')
explosion_1 = pygame.transform.scale(explosion_1, (150, 150))
explosion_2 = load_image('explosion_2.tiff')
explosion_2 = pygame.transform.scale(explosion_2, (150, 150))

# loading demon images

# demon_1 = load_image('demon_1.png')
# demon_2 = load_image('demon_2.png')
# demon_3 = load_image('demon_3.png')
# demon_4 = load_image('demon_4.png')
# demons = (demon_1, demon_2, demon_3, demon_3)

# loading demon images

coin_1 = load_image('sonicring-1.png')
coin_2 = load_image('sonicring-2.png')
coin_3 = load_image('sonicring-3.png')
coin_4 = load_image('sonicring-4.png')

coins = [coin_1, coin_2, coin_3, coin_4]

# loading demon images

planet_1 = load_image('planet_1.tiff')
planet_2 = load_image('planet_2.tiff')
planet_3 = load_image('planet_3.tiff')
planet_4 = load_image('planet_4.tiff')
planet_5 = load_image('planet_5.tiff')
planet_6 = load_image('planet_6.tiff')
planet_7 = load_image('planet_7.tiff')
planet_8 = load_image('planet_8.tiff')

planets = [planet_1, planet_2, planet_3, planet_4, planet_5, planet_6, planet_7, planet_8]

# loading title image

title_1 = load_image('title_1.png')
title_2 = load_image('title_2.png')
new_game = load_image('new_game.png')
load_game = load_image('load_game.png')
settings = load_image('settings.png')
credits = load_image('credits.png')

# intro_loop()
menu_loop()
game_loop()
pygame.quit()
quit()
