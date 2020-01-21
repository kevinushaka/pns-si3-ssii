# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""

import matplotlib.pyplot as plt
import numpy as np
import librosa.display
import librosa
import sounddevice as sd

x, fe = librosa.load('music.wav')
plt.figure(figsize=(14, 5))
librosa.display.waveplot(x, sr=fe)
librosa.load('music.wav', mono=False)
plt.title('doM')
plt.show()
n=len(x)
t = np.linspace(0, n/fe, n, endpoint=False)
plt.plot(t,x)

sd.play(x,fe)
status = sd.wait()