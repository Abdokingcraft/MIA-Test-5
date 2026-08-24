# An Audio Journey

[The code](#the-code)

At first when I heard the audio at 100% Volume my ears went **~~ÂÂÁÃÅÀ~~** could have at least left an audio warning at the beginning or even a text when sending it

After that I checked the spectrogram with audacity it was full to the sky

The first thing I noticed is the horizontal line and I said "that must me it, I just have to remove them" I didn't know about a library that would do that so I removed it manually by audacity first (they were a lot of them)

And after I removed them it did .... Absolutely nothing to the audio 

So after that I was thinking of how would a normal spectrogram look like, so i made an audio my self and I noticed 3 things

1- the audio isn't connected to eachother ( every word have spaces between them

2- the frequency was in a range (between 400 and 4000) and wasn't in the 12k 💀

3- nothing in my speech wasn't as bright as the audio

So after I know all of that I began to research 

# The code

#### LOOK AT THE IPYNB FOR SPECTROGRAM VISUALIZATION

**Stage 1: HPSS**

```python
harmonic, percussive = librosa.effects.hpss(audio, margin=2.0)
y_work = percussive
```

Splits the audio into "tonal" (harmonic) and "sharp/noisy" (percussive) parts. Keeps only the percussive part (Speech)

**Stage 2: Spectral Gating**

```python
threshold = np.percentile(mag, 50)
mag_gated = np.where(mag > threshold * 1.5, mag, mag * 0.1)
```

Looks at loudness across the frequency spectrum and quiets down anything below a threshold. Assumes quiet = probably noise, loud = probably speech.

**Stage 3: Bandpass Filter**

```python
b, a = signal.butter(4, [low, high], btype='band')
y_work = signal.filtfilt(b, a, y_work)
```

Keeps only frequencies between 300–4000 Hz, since that's the range human speech lives in. Everything outside that range (rumble, hiss) gets cut.

**Stage 4: Spectral Subtraction**

```python
noise_profile = np.mean(mag[:, :5], axis=1, keepdims=True)
mag_sub = np.maximum(0, mag - (2.0 * noise_profile))
```

Samples the first few frames as a "noise fingerprint," then subtracts that fingerprint from the whole signal. Removes noise that's steady throughout the recording.

**Stage 5: Trimming**

```python
y_work = y_work[int(sr * 0.03):int(sr * 0.45)]
```

Cuts the audio down to just the 0.03s–0.45s window where the actual speech is, dropping everything else.


