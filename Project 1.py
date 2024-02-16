#import necessary modules
import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import IPython.display as ipd
#question 1 pt 1, recording audio
# Define Sampling Rate or Frequency in Hz
sr = 44100
# Record duration in seconds
duration = 6
# Start audio recording
#recording = sd.rec(int(duration*sr), samplerate=sr, channels=2) 
# we will record with a  mono or stereo channel microphone

# Record audio for the given duration
#print("recording...............")
#sd.wait()

# Write it to a file
#write("sound3.wav",sr,recording)
#question 1 pt 2, plotting the graphs + question 3, spectrographs
#loading audio 1

x1, sr = librosa.load("sound1.wav")
print("recording shape", x1.shape)
print("sampling rate", sr)
#printing graph of audio 1
librosa.display.waveshow(x1, sr=sr, color="blue")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.title("Sound Waveform 1")
plt.show()
#printing spectrogram of audio 1
freq_in_db = librosa.amplitude_to_db(np.abs(librosa.stft(x1)), ref=8000)
plt.figure(figsize=(8, 6))
librosa.display.specshow(freq_in_db, sr=sr)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Sound Spectrogram 1")
plt.show()
#loading audio 2
x2, sr = librosa.load("sound2.wav")
print("recording shape", x2.shape)
print("sampling rate", sr)
#printing graph of audio 1
librosa.display.waveshow(x2, sr=sr, color="blue")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.title("Sound Waveform 2")
plt.show()
#printing spectrogram of audio 2
freq_in_db = librosa.amplitude_to_db(np.abs(librosa.stft(x2)), ref=8000)
plt.figure(figsize=(8, 6))
librosa.display.specshow(freq_in_db, sr=sr)
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.title("Sound Spectrogram 2")
plt.show()
#loading sound 3
x3, sr = librosa.load("sound3.wav")
print("recording shape", x3.shape)
print("sampling rate", sr)
#displaying soundwave 3 in normal graph
librosa.display.waveshow(x3, sr=sr, color="blue")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.title("Sound Waveform 3")
plt.show()
#displaying soundwave 3 in spectogram
freq_in_db = librosa.amplitude_to_db(np.abs(librosa.stft(x3)), ref=8000)
plt.figure(figsize=(8, 6))
librosa.display.specshow(freq_in_db, sr=sr)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Sound Spectrogram 3")
plt.show()

##Question 6, Time Delay
x2, sr = librosa.load("sound2.wav")
con_left = np.concatenate([np.zeros(6), x2[0:-6]])
write("sound2.wav", sr, np.array(x2))
#0ms
con_left = np.concatenate([np.zeros(6), x2[0:-6]])
ar = np.array([x2,con_left])
write("sound2_0ms.wav", sr, ar.T)
#1ms
con_left = np.concatenate([np.zeros((int(sr*0.001))), x2[0:-(int(sr*0.001))]])
ar = np.array([x2,con_left])
write("sound2_1ms.wav", sr, ar.T)
#10ms
con_left = np.concatenate([np.zeros((int(sr*0.01))), x2[0:-(int(sr*0.01))]])
ar = np.array([x2,con_left])
write("sound2_10ms.wav", sr, ar.T)
#100ms
con_left = np.concatenate([np.zeros((int(sr*0.1))), x2[0:-(int(sr*0.1))]]) 
ar = np.array([x2,con_left])
write("sound2_100ms.wav", sr, ar.T)
#head
con_left = np.concatenate([np.zeros((int(sr*0.000518))), x2[0:-(int(sr*0.000518))]])
ar = np.array([x2,con_left])
write("sound2_avghead.wav", sr, ar.T)

##Question 7 Attenuation
#0ms
#1.5
con_left = np.concatenate([np.zeros(6), x2[0:-6]]) * 0.75
ar = np.array([x2,con_left])
write("0ms-1.5db.wav", sr, ar.T)
#3
con_left = np.concatenate([np.zeros(6), x2[0:-6]]) * 0.5
ar = np.array([x2,con_left])
write("0ms-3db.wav", sr, ar.T)
#6
con_left = np.concatenate([np.zeros(6), x2[0:-6]]) * 0.25
ar = np.array([x2,con_left])
write("0ms-6db.wav", sr, ar.T)
#head
#1.5
con_left = np.concatenate([np.zeros((int(sr*0.000518))), x2[0:-(int(sr*0.000518))]])*0.75
ar = np.array([x2,con_left])
write("518us-1.5db.wav", sr, ar.T)
#3
con_left = np.concatenate([np.zeros((int(sr*0.000518))), x2[0:-(int(sr*0.000518))]])*0.5
ar = np.array([x2,con_left])
write("518us-3db.wav", sr, ar.T)
#6
con_left = np.concatenate([np.zeros((int(sr*0.000518))), x2[0:-(int(sr*0.000518))]])*0.25
ar = np.array([x2,con_left])
write("518us-6db.wav", sr, ar.T)
