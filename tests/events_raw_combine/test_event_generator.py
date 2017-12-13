from pynfb.generators import stream_generator_in_a_thread, run_events_sim
from pynfb.inlets.lsl_inlet import LSLInlet
import time

if __name__ == '__main__':
    stream_generator_in_a_thread('test_events', run_events_sim)
    stream_generator_in_a_thread('NVX136_Data')
    print('started')
    inlet = LSLInlet('test_events')
    while True:
        time.sleep(0.5)
        print(*inlet.get_next_chunk())