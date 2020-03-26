# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 10:09:21 2020

@author: Kévin
"""

import matplotlib.pyplot as plt
import numpy as np
import librosa.display
import librosa
import sounddevice as sd
import matplotlib.pyplot as plt
from scipy import signal

# Generate the time vector properly
x, fe = librosa.load('ressources/PIANO.wav')
plt.figure(figsize=(14, 5))
librosa.display.waveplot(x, sr=fe)
n=len(x)
t = np.linspace(0, n/fe, n, endpoint=False)
s = 0.75*np.cos(2*np.pi*440*t) 
plt.show()

fc = 30  # Cut-off frequency of the filter
w = fc / (fe / 2) # Normalize the frequency
b, a = signal.butter(5, w, 'low')
output = signal.filtfilt(b, a, s)
plt.plot(t, output, label='filtered')
plt.legend()
plt.show()

#plt.plot(np.abs(np.fft.fft(s)));
plt.plot(np.abs(np.fft.fft(output)));

sd.play(x,fe)