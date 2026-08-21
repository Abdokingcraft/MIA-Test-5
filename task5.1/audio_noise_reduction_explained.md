# Audio Noise Reduction 

## The Problem

When I looked at the spectrogram of the original audio, I saw a straight horizontal line running across it. My first thought was: that's the noise, that's the rule — just filter that line out.

I looked for a filter that could remove it directly but couldn't find one, so I tried doing it manually in Audacity instead. While doing that, I realized the horizontal line wasn't the whole problem — it was only *part* of the noise. That sent me looking for a proper filtering pipeline instead of a single fix.

## Stage 1: Harmonic-Percussive Source Separation (HPSS)

This was the filter I eventually found that could deal with the horizontal-line noise automatically, instead of me erasing it by hand in Audacity.

**In simple terms:** HPSS splits audio into two parts — the *harmonic* part (smooth, sustained tones, like a hum or a musical note) and the *percussive* part (short, sharp bursts, like speech consonants and transients). Since the noise I was seeing behaved like a steady tone, it lived mostly in the harmonic part. Speech, on the other hand, has a lot of percussive character. So I kept the percussive component and used that going forward.

## Stage 2: Aggressive Spectral Gating

After stage 1, the horizontal line was better, but I noticed the leftover noise now showed up differently — not as one clean horizontal line anymore, but as scattered vertical lines across the spectrogram.

**In simple terms:** Spectral gating looks at the audio's spectrogram (a picture of which frequencies are present at each moment) and compares every point to a threshold — I used the 50th percentile of the signal's magnitude, then made it aggressive by requiring anything below 1.5× that threshold to get pushed way down (kept at only 10% of its strength). So quiet, noise-like content gets suppressed while louder, meaningful content survives. This is what helped with those vertical-line artifacts left over from stage 1.

## Stage 3: Bandpass Filter (300 Hz – 4000 Hz)

I looked at a normal spectrogram of my own voice and noticed the frequency content didn't go very high — even when I raised my voice, it topped out around 4000 Hz. I also noticed that frequencies outside that range didn't connect meaningfully to the rest of the speech across the earlier stages.

**In simple terms:** Human speech mostly lives between roughly 300 Hz and 4000 Hz. A bandpass filter only lets frequencies inside that range through, and cuts everything above and below it. Since I knew my voice doesn't go above ~4000 Hz, anything outside 300–4000 Hz was very unlikely to be my speech — so it was safe to filter out.

## Stage 4: Spectral Subtraction

This was the stage I didn't fully understand yet — it felt like "actual magic," working against noise itself rather than against a fixed range like the bandpass filter.

**In simple terms:** Spectral subtraction works by first estimating what the *noise alone* sounds like — I did this by sampling the very first few frames of audio (before real speech starts, when it should be mostly just background noise) and averaging their frequency content into a "noise profile." Then, for the *entire* audio, that noise profile gets subtracted out of every frame's frequency content. Whatever is left over is treated as the "real" signal (the speech), with the estimated noise stripped away. It's the only stage that actively *measures* the noise from the recording itself and removes exactly that shape — the other stages just apply general rules (harmonic vs. percussive, a loudness threshold, a fixed frequency range) without ever looking at what the actual noise in *this* recording looked like.

## Stage 5: Trimming

Finally, the very start and end of the audio file had extra amplification/noise that wasn't part of the actual speech. So the last step was simply trimming the file down to keep only the real speech portion in the middle, cutting off the noisy start and end.

## Summary

| Stage | Technique | What it targets |
|---|---|---|
| 1 | Harmonic-Percussive Separation | Steady, tone-like background hum |
| 2 | Aggressive Spectral Gating | Leftover scattered noise (quiet content) |
| 3 | Bandpass Filter (300–4000 Hz) | Frequencies outside the human speech range |
| 4 | Spectral Subtraction | Noise pattern learned directly from the recording itself |
| 5 | Trimming | Extra amplified noise at the very start/end of the file |

Each stage tackled a different kind of leftover noise the previous stage didn't fully catch — going from a manual, single "erase the obvious line" fix to a full multi-stage pipeline.
