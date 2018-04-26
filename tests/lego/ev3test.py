from time import sleep
import rpyc
with rpyc.classic.connect('ev3dev') as a:
    ev3 = a.modules['ev3dev.ev3']
    m = ev3.LargeMotor('outB')
    m.run_timed(time_sp=1000, speed_sp=-100)
    sleep(5)
    m.run_to_abs_pos(position_sp=0, speed_sp=100,  stop_action="hold")