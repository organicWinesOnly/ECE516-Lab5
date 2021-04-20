""" This file contains a function to capture audio from the connected
    microphone.

    Not used in this lab.
"""

import pyaudio
from numpy import fromstring

def capture(stream):
    """ capture new audio from microphone and covert in to np array of floating
        point numbers.

        ==== Examples ==== 
        >>> player = pyaudio.PyAudio()
        >>> stream = player.open(format=pyaudio.paFloat32,
                                 channels=1,
                                 input=True,
                                 rate=fs,
                                 ouput=True)
        >>> data = capture(stream)
        >>> stream.stop_stream()
        >>> stream.close()
        >>> player.terminate()
    """
    data = stream.read()
    data = fromstring(data, 'Float32')
    return data
