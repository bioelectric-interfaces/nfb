from lsl_inlet import LSLInlet
from convert_to_fif import save_fif
from time import time
import numpy as np
import pygame
import os


# edit only these settings if necessary:
stream_name = 'Mitsar'
sound_file = r'med_instr.mp3'


# connect to stream
inlet = LSLInlet(stream_name)
fs = int(inlet.get_frequency())
channels = inlet.get_channels_labels()
print('Connected to {}\nFs: {}\nChannels: {}'.format(stream_name, fs, channels))

# check first second data
t0 = time()
data_counter = 0
while time() - t0 < 1:
    data_counter += inlet.get_next_chunk()[0] is not None
if data_counter == 0:
    raise ConnectionError('First second is empty. Try to reconnect')

# set and create new directory
record_dir = input('Write experiment directory name:\n')
if not os.path.exists(record_dir):
    os.makedirs(record_dir)
else:
    raise FileExistsError('Recording directory exists')

# save montage
with open('{}\montage.info'.format(record_dir), 'w', encoding="utf-8") as f:
    f.write(' '.join([str(fs)]+channels + ['timestamps']))

# setup buffer
buffer_size = int(fs * 15)
samples_counter = 0
buffers_counter = 1
buffer = np.zeros((buffer_size, len(channels) + 1)) * np.nan

# load sound
pygame.mixer.init()
pygame.mixer.music.load(sound_file)
clock = pygame.time.Clock()

# run experiment
input('Press any key to start...')
t0 = time()
pygame.mixer.music.play()
print('Experiment is ON AIR!')
while pygame.mixer.music.get_busy():
    clock.tick(2)
    chunk, timestamp = inlet.get_next_chunk()
    if chunk is not None:
        chunk_size = chunk.shape[0]
        if samples_counter + chunk_size > buffer_size:
            np.save('{}/record_{}.npy'.format(record_dir, buffers_counter), buffer[np.isfinite(buffer[:,0])])
            print('{:02d}:{:02d}. Buffer {} was saved'.format(int(time()-t0)//60, int(time()-t0)%60, buffers_counter))
            buffer *= np.nan
            buffers_counter += 1
            samples_counter = 0
        buffer[samples_counter:samples_counter + chunk_size, :-1] = chunk
        buffer[samples_counter:samples_counter + chunk_size, -1] = timestamp
        samples_counter += chunk_size

np.save('{}/record_{}.npy'.format(record_dir, buffers_counter), buffer[np.isfinite(buffer[:,0])])
print('Done!')

save_fif(record_dir)
print('Records was converted to .fif')