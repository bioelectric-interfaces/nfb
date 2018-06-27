from pynfb.inlets.lsl_inlet import LSLInlet
from time import sleep, time
import rpyc



# connect to ev3
a = rpyc.classic.connect('ev3dev')
ev3 = a.modules['ev3dev.ev3']

# connect to motor
motor = ev3.LargeMotor('outB')
ev3.Leds.set_color(ev3.Leds.LEFT, ev3.Leds.GREEN)
ev3.Leds.set_color(ev3.Leds.RIGHT, ev3.Leds.GREEN)

motor.run_to_abs_pos(position_sp=0, speed_sp=100, stop_action="hold")
sleep(3)

# connect to nfb
inlet = LSLInlet('NFBLab_data')
t = time()
while True:
    sleep(0.6)
    chunk, timestamp = inlet.get_next_chunk()
    if chunk is not None:
        print(chunk[:, 0].mean())
        if chunk[:, 0].mean() < 1.2:
            t = time()
            print('RIGHT!!!!!!!!!!!!!!!!!!!!!!!!')
            motor.run_forever(speed_sp=+20)
            ev3.Leds.set_color(ev3.Leds.LEFT, ev3.Leds.GREEN)
            ev3.Leds.set_color(ev3.Leds.RIGHT, ev3.Leds.GREEN)
        else:
            motor.stop()
            ev3.Leds.set_color(ev3.Leds.LEFT, ev3.Leds.RED)
            if time() - t > 3:
                ev3.Leds.set_color(ev3.Leds.RIGHT, ev3.Leds.RED)
                motor.run_to_abs_pos(position_sp=0, speed_sp=100, stop_action="hold")
                sleep(5)
                t = time()



        # elif chunk[:, 0].mean() > 1.5 and chunk[:, 1].mean() < 1.5:
        #     print('LEFT!!!!!!!!!!!!!!!!!!!!!!!!')
        #     motor.run_timed(time_sp=100, speed_sp=+100)