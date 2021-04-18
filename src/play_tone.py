""" Function to play a tone of a certain frequency.
"""

import pyaudio
import numpy as np

def msin(freq, fs, samples):
    """ Generate a sine wave with <freq> Hz sampled at <fs> Hz of length <samples>.
    """
    times = np.arange(samples) * freq / fs
    return np.sin(2 * np.pi * times)


def playTone(volume, freq, fs, samples):
    """ Play a tone of a with <freq> Hz sampled at <fs> Hz of length <samples>.
        To specify the duration in seconds set <samples> = <fs> * duration.
        
        === params ===
        volume - between 0 and 1
    """
    signal = (volume * msin(freq, fs, samples)).astype(np.float32)
    # Convert float to bytes
    sig_bytes = signal.tobytes()
    
    # build sound environment
    player = pyaudio.PyAudio()
    stream = player.open(format=pyaudio.paFloat32, 
                         channels=1,
                         rate=fs,
                         output=True)
    # play tone
    stream.write(sig_bytes)
    # close environment
    stream.stop_stream()
    stream.close()
    player.terminate()


def playGivenTone(volume, sound, fs):
    """ Play a sound stored as an np.ndarry <sound> sampled at <fs> Hz.
        
        === params ===
        volume - between 0 and 1
    """
    signal = (volume * sound).astype(np.float32)
    # Convert float to bytes
    sig_bytes = signal.tobytes()
    
    # build sound environment
    player = pyaudio.PyAudio()
    stream = player.open(format=pyaudio.paFloat32, 
                         channels=1,
                         rate=fs,
                         output=True)
    # play tone
    stream.write(sig_bytes)
    # close environment
    stream.stop_stream()
    stream.close()
    player.terminate()
