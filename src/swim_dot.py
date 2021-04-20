""" Soft implementation of a lock-in amplifier and Sequential Wave Imprinting
    Machine (SWIM).

    This version displays a moving dot that only tranverses the y direction
    and is suitible for wave  mapping using long exposure photographs.
"""
from play_tone import msin
from capture import capture
from filters import lowpass_filter
import pyaudio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as amt
import sys

# Constants
DURATION = float(sys.argv[1])
FRAME_RATE = 24
NUM_FRAME = 512 # 1024
GAIN = 25
fs = 40000

# =============================================================================
# Set up plotting environment
# =============================================================================
# Various plotting colours
dgreen = np.array([0,58,23]) / 255
green = np.array([40, 254, 170]) / 255
white = np.array([253,255,254]) / 255

# Matplotlib figure attributes
fig, ax = plt.subplots(figsize=(16,8), 
                       facecolor=(0,0,0)) 

ax.set_facecolor((0,0,0))
ax.set_xlim(0, NUM_FRAME) 
ax.set_position([0, 0,  1, 1])

# =============================================================================
# Lock-in amplifier function
# =============================================================================
def lockIn(ref, recorded):
    assert (ref.size == recorded.size)
    # Multiply signals
    prod = ref * recorded
    # filter result and return
    return lowpass_filter(prod, fs, limit=40) * GAIN

# =============================================================================
# Set up microphone stream 
# =============================================================================
# generate pure tone as reference
tone = msin(500, fs, fs * DURATION)
# initialize container to store lock-in amplifier results
output = np.zeros(NUM_FRAME)


i = 0
def callback(input_data, frame_count, time, status):
    """ Function runs as an intermediate processing step.
    """
    data = np.frombuffer(input_data, np.float32)
    global output 
    global i 
    if (i + 1) * frame_count >= tone.size:
        i = 0
    output = lockIn(tone[i * frame_count:(i+1) * frame_count], data)
    i += 1
    output_bytes = output.astype(np.float32).tobytes() 
    return (output_bytes, pyaudio.paContinue)


player = pyaudio.PyAudio()
stream = player.open(format=pyaudio.paFloat32,
                     channels=1,
                     input=True,
                     rate=fs,
                     frames_per_buffer=NUM_FRAME,
                     stream_callback=callback)

# start microphone recording
stream.start_stream()

# =============================================================================
# Initialize plots and animation intermediate function
# =============================================================================
l, = plt.plot([NUM_FRAME //2], output[NUM_FRAME //2], 'o', 
              c=dgreen, lw=6, alpha=0.6)
l1, = plt.plot([NUM_FRAME //2], output[NUM_FRAME //2],'o', 
               c=green, lw=4, alpha=0.7)
l2, = plt.plot([NUM_FRAME //2], output[NUM_FRAME //2],'o', 
               c=white, lw=2, alpha = 0.1)

def FrameDeveloper(i):
    """ Update the frame for the animation.
    """
    global output
    l.set_data([NUM_FRAME //2], output[NUM_FRAME //2])
    l2.set_data([NUM_FRAME //2], output[NUM_FRAME //2])
    l1.set_data([NUM_FRAME //2], output[NUM_FRAME //2])
    return l, l1, l2

# =============================================================================
# Build and run animation
# =============================================================================
ani = amt.FuncAnimation(fig, FrameDeveloper, 
                        frames=int(FRAME_RATE * DURATION), 
                        interval=1000/FRAME_RATE,
                        blit=True)

plt.show()

# =============================================================================
# Close stream
# =============================================================================
stream.stop_stream()
stream.close()
player.terminate()
