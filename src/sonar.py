""" Soft implementation of a lock in amplifier and a Sequential Wave
    imprinting Machine (SWIM) for sonar.

    The program outputs a plot of the first band from the time
    frequency diagram of the picked up sound. 

    A q-chirplet (quadratic) transform is used for the program.
"""
from play_tone import msin, mcos
from capture import capture
from filters import lowpass_filter
from polynomial_chirp_trans import PolynomialCT 
import pyaudio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as amt
import sys

# Constants
DURATION = float(sys.argv[1])
FRAME_RATE = 60
NUM_FRAME = 1024 * 4
GAIN = 5
fs = 40000
# Quadratic chirplet transform parameters
WIN_SIZE=256
OVERLAP=128

# =============================================================================
# Set up plotting environment
# =============================================================================
# Various plotting colours
dgreen = np.array([0,58,23]) / 255
green = np.array([40, 254, 170]) / 255
white = np.array([253,255,254]) / 255

# Matplotlib figure attributes
fig, ax = plt.subplots(figsize=(16,8), facecolor=(0,0,0))
ax.set_facecolor((0,0,0))
ax.set_position([0, 0,  1, 1])

# =============================================================================
# Lock-in amplifier function
# =============================================================================
def lockIn(ref, recorded):
    """ Lock in amplifier.
    """
    assert (ref.size == recorded.size)
    # Multiply signals
    prod = ref * recorded
    # filter result and return
    return lowpass_filter(prod, fs, limit=40) * GAIN

# =============================================================================
# Set up microphone stream 
# =============================================================================
# generate pure sine tone as reference
tone = msin(500, fs, fs * DURATION)
# initialize container to store lock-in amplifier results
output = lowpass_filter(tone[:NUM_FRAME], fs, limit=40) * GAIN
pct = PolynomialCT(output, fs, WIN_SIZE, OVERLAP, poly_order=2)

try:
    pct.Run(0.001, 5)
except ValueError:
    pass
output = pct.tfd[1, :]


i = 0
def callback(input_data, frame_count, time, status):
    """ Function runs as an intermediate processing step.
    """
    data = np.frombuffer(input_data, np.float32)
    global output
    global i 
    if (i + 1) * frame_count >= tone.size:
        i = 0
    output = lockIn(tone[i * frame_count:(i+1) * frame_count], 
                    data)
    i += 1
    output_bytes = output.astype(np.float32).tobytes() 
    return (output_bytes, pyaudio.paContinue)


# Initialize microphone
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
l, = plt.plot(np.arange(output.size), np.abs(output), c=dgreen, lw=6, alpha=0.6)
l1, = plt.plot(np.arange(output.size), np.abs(output), c=green, lw=4, alpha=0.7)
l2, = plt.plot(np.arange(output.size), np.abs(output), c=white, lw=2, alpha = 0.8)


def FrameDeveloper(i):
    """ Update the frame for the animation.
    """
    global output
    pct = PolynomialCT(output, fs, WIN_SIZE, OVERLAP, poly_order=2)
    try:
        pct.Run(0.001, 5)
    except ValueError:
        pass
    output = pct.tfd[1, :]
    l.set_data(np.arange(output.size), np.abs(output))
    l1.set_data(np.arange(output.size), np.abs(output))
    l2.set_data(np.arange(output.size), np.abs(output))
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
