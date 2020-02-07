# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 10:52:42 2020

@author: Kévin
"""

import matplotlib.pyplot as plt
import numpy as np
import librosa.display
import librosa
import sounddevice as sd

"""
N=16000
fe=8000
t=np.linspace(0,N/fe,N);
s = 0.2*np.cos(2*np.pi*200*t) + 2*np.cos(2*np.pi*400*t);
tf=np.linspace(0,fe/N,N);
plt.subplot(1,2,1);
plt.plot(t[:200],s[:200]);
plt.title('280Hz et 500Hz,fe=8000Hz')
plt.subplot(1,2,2);
plt.plot(np.abs(np.fft.fft(s)));
plt.title('280Hz et 500Hz,fe=8000Hz')
"""
#x, fe = librosa.load('ressources/mesange-tete-noire.wav')
x, fe = librosa.load('ressources/PIANO.wav')
plt.figure(figsize=(14, 5))
librosa.display.waveplot(x, sr=fe)
plt.title('')
plt.show()
fe/=16
n=len(x)
t = np.linspace(0, n/fe, n, endpoint=False)
s = 0.75*np.cos(2*np.pi*440*t) 
plt.plot(t,x)
plt.plot(np.abs(np.fft.fft(s)));
Sdb = librosa.amplitude_to_db(abs(s))

S = np.abs(librosa.stft(s))
Sdb = librosa.amplitude_to_db(abs(S))
#librosa.display.specshow(Sdb, sr=fe, x_axis='time', y_axis='hz')
#librosa.display.specshow(Sdb, sr=fe, x_axis='time', y_axis='hz')

sd.play(x,fe)
status = sd.wait()