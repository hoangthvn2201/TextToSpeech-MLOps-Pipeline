# Course 2: Digital Signal Processing for Audio

**Prerequisite:** Course 1 (Python for ML Engineers). NumPy basics (arrays, indexing, `np.mean`).
**Goal:** After this course you can read `audio_processor.py`, `feature_extractor.py`, and `audio_utils.py` completely — every parameter has a reason you can explain.

---

## Module 1 — What Audio Actually Is

Before writing any code, you need a physical mental model. Everything else is derived from this.

### 1.1 Sound as pressure waves

Sound is air pressure changing over time. When you speak, your vocal cords vibrate and push air molecules together and apart in waves. A microphone converts those pressure changes into a continuously varying electrical voltage — an **analog signal**.

The key properties of any audio signal:

| Property | What it is | Unit |
|---|---|---|
| **Amplitude** | How strong the pressure change is | Dimensionless (normalized −1.0 to +1.0 in digital) |
| **Frequency** | How many pressure cycles per second | Hz (Hertz) |
| **Phase** | Where in the cycle the wave starts | Radians |

### 1.2 From analog to digital: sampling

A computer cannot store a continuously varying signal. It takes **snapshots** of the amplitude at regular intervals. Each snapshot is called a **sample**. The rate of snapshots is the **sample rate** (or sampling rate).

```
Analog signal (continuous):

  amplitude
    │ ╭──╮      ╭──╮
    │╭╯  ╰╮    ╭╯  ╰╮
    ││    ╰╮  ╭╯    ╰──
    ╰╯     ╰──╯
         ──────────────▶ time

Digital samples (discrete):

  amplitude
    │ ·  ·          ← samples at fixed intervals
    │·    ·
    │      · ·    ·
    │         · ·
         ──────────────▶ time (sample index)
```

The result is a 1D array of floats. This is what `torchaudio.load()` returns:

```python
import torchaudio

waveform, sr = torchaudio.load("speech.wav")
# waveform: Tensor of shape (channels, num_samples)
# sr: sample rate, e.g. 22050
```

### 1.3 Bit depth and dynamic range

Each sample is stored as an integer of fixed width:

| Bit depth | Possible values | Dynamic range |
|---|---|---|
| 8-bit | 256 levels | 48 dB |
| 16-bit | 65,536 levels | 96 dB |
| 24-bit | 16,777,216 levels | 144 dB |

This project saves normalized audio as **16-bit PCM** ([audio_processor.py:47](../src/tts_pipeline/data/audio_processor.py#L47)):
```python
torchaudio.save(str(dst), waveform, target_sr, encoding="PCM_S", bits_per_sample=16)
```

16-bit gives 96 dB of dynamic range — more than enough for speech (which spans roughly 40–60 dB).

---

## Module 2 — Sample Rate and the Nyquist Theorem

### 2.1 Why 22050 Hz

This project uses `sample_rate: 22050` ([params.yaml:7](../params.yaml#L7)). This is not arbitrary.

The human voice produces frequencies from roughly 80 Hz (low male vocal fry) to 8000 Hz (high-frequency consonants like /s/ and /sh/). Human hearing extends to ~20,000 Hz, but **speech intelligibility** is largely captured below 8000 Hz.

### 2.2 The Nyquist theorem

**Theorem:** To accurately represent a signal with maximum frequency `f_max`, the sample rate must be at least `2 × f_max`.

```
Nyquist frequency = sample_rate / 2
```

At 22050 Hz:
```
Nyquist frequency = 22050 / 2 = 11025 Hz
```

This means: any frequency below 11025 Hz is faithfully captured. Since speech energy above 8000 Hz is minimal and `f_max = 8000` in this project ([params.yaml:13](../params.yaml#L13)), 22050 Hz captures everything that matters for speech.

Why not use 44100 Hz (CD quality)? You *could*, but:
- Doubles memory usage (twice as many samples per second)
- Doubles compute for every downstream operation (FFT, convolution)
- Adds no useful information for speech

Why not 16000 Hz (common in telephony ASR)?
- Nyquist limit = 8000 Hz — just barely captures `/s/` and `/sh/`
- 22050 Hz gives more headroom and better consonant quality

### 2.3 Aliasing — what happens when you violate Nyquist

If you sample a 10000 Hz tone at 8000 Hz, you violate Nyquist. The samples are taken too slowly to see two full cycles. The reconstructed signal *looks like* a 2000 Hz tone — a frequency that was never in the original. This is **aliasing**.

In practice: resampling algorithms include a **low-pass filter** before downsampling to cut frequencies above the new Nyquist limit. `torchaudio.transforms.Resample` does this automatically.

### 2.4 Resampling in code

```python
# audio_processor.py:31-32
if sr != target_sr:
    waveform = torchaudio.transforms.Resample(sr, target_sr)(waveform)
```

If source audio is at 44100 Hz and target is 22050 Hz, this:
1. Applies a low-pass filter at 11025 Hz (new Nyquist)
2. Takes every other sample

The result is half as many samples, same duration, no aliasing.

---

## Module 3 — The Fourier Transform and FFT

### 3.1 The fundamental insight

A waveform in the time domain tells you "what the amplitude was at each moment." But it does not directly tell you "which frequencies are present."

**The Fourier Transform converts a time-domain signal into a frequency-domain representation.** It answers: "what mix of pure sine waves, at what amplitudes, would produce this waveform?"

```
Time domain:                     Frequency domain:
                                 amplitude
  amplitude                        │   █
    │ ╭╮╭╮╭╮╭╮╭╮                   │   █
    │╭╯╰╯╰╯╰╯╰╯╰╮  ── FFT ──▶      │   █  █
    ╰╯           ╰╯                 │   █  █  █
         ──────────▶ time           └──────────▶ frequency
                                        100 440 880 Hz
```

### 3.2 Frequency bins

The FFT of an `n_fft`-point signal produces `n_fft // 2 + 1` **frequency bins** (for real-valued signals). Each bin corresponds to a frequency:

```
bin_frequency[k] = k * (sample_rate / n_fft)
```

With `n_fft = 1024` and `sample_rate = 22050`:

```
bin 0   → 0 Hz        (DC component, average value)
bin 1   → 21.5 Hz
bin 2   → 43.0 Hz
...
bin 512 → 11025 Hz    (Nyquist — highest bin)
```

Total bins: `1024 // 2 + 1 = 513`

**Frequency resolution** = `sample_rate / n_fft = 22050 / 1024 ≈ 21.5 Hz per bin`

This is the trade-off: larger `n_fft` → finer frequency resolution, but requires more samples (longer time window) per FFT.

### 3.3 Time-frequency trade-off

This is fundamental and comes up every time you tune mel spectrogram parameters:

```
Larger n_fft → better frequency resolution, worse time resolution
Smaller n_fft → better time resolution, worse frequency resolution
```

For speech, this matters. Fast consonants (/t/, /p/, /k/) need fine time resolution. Vowels (slow, periodic) benefit from fine frequency resolution. 1024 at 22050 Hz is a well-tested compromise for TTS.

---

## Module 4 — STFT: Short-Time Fourier Transform

### 4.1 The problem with a single FFT

If you apply FFT to an entire 3-second speech recording, you get one frequency snapshot for the whole 3 seconds. But speech changes every ~20ms — the phoneme /a/ has a completely different spectrum from /s/. A global FFT merges them all.

**Solution:** divide the signal into short overlapping windows, apply FFT to each window. This is the **Short-Time Fourier Transform (STFT)**.

### 4.2 Three parameters that define the STFT

All three appear directly in `DataConfig` and `params.yaml`:

| Parameter | Value | Meaning |
|---|---|---|
| `n_fft` | 1024 | FFT size — controls frequency resolution |
| `win_length` | 1024 | Window size — how many samples per window |
| `hop_length` | 256 | Step size between windows |

**Window:** A chunk of `win_length` samples extracted from the signal.
**Hop:** Each new window starts `hop_length` samples after the previous one.
**Overlap:** `win_length - hop_length = 1024 - 256 = 768` samples of overlap.

```
Signal (samples):  |--sample_0--sample_1--sample_2--...--|

Window 1:          [====window (1024 samples)====]
Window 2:                   [====window====]        (shifted by 256)
Window 3:                            [====window====]
Window 4:                                     [====window====]
```

**How many frames?** For a signal of `N` samples:
```
n_frames ≈ N / hop_length
```

For a 1-second clip at 22050 Hz:
```
n_frames = 22050 / 256 ≈ 86 frames
```

So a 1-second clip becomes a (513 freq_bins × 86 time_frames) matrix.

### 4.3 The Hann window — why we window at all

A sharp rectangular cutout of a signal (just taking samples 0–1023 with no taper) creates **spectral leakage**: energy from one frequency "bleeds" into neighboring bins because the abrupt edges create discontinuities.

The **Hann window** is a smooth taper — it multiplies the signal samples by a raised cosine before the FFT:

```
w[n] = 0.5 * (1 - cos(2π * n / (win_length - 1)))
```

This means samples at the center of the window are weighted fully; samples at the edges smoothly fade to zero. Result: much less spectral leakage.

`torchaudio.transforms.MelSpectrogram` uses a Hann window by default.

### 4.4 `center=True` and `pad_mode="reflect"`

```python
# feature_extractor.py:38-39
center=True,
pad_mode="reflect",
```

When `center=True`, the signal is padded so that the `t`-th frame is centered on sample `t * hop_length`. Without padding, the first frame would start at sample 0 and miss the beginning of the signal.

`pad_mode="reflect"` pads by reflecting the signal at the edges (mirroring), which avoids introducing artificial silence. Cleaner than zero-padding for edge frames.

---

## Module 5 — Mel Scale and Mel Filterbank

### 5.1 The non-linearity of human hearing

Human perception of pitch is not linear in frequency. The difference between 100 Hz and 200 Hz sounds much larger than the difference between 5000 Hz and 5100 Hz, even though both are 100 Hz gaps.

This means: a raw FFT spectrogram with 513 linearly-spaced bins from 0–11025 Hz wastes most of its bins on high frequencies where humans have poor pitch resolution, and under-represents low frequencies where humans are most sensitive.

### 5.2 The Mel scale

The **Mel scale** is a perceptual frequency scale that matches human pitch perception:

```
mel(f) = 2595 * log10(1 + f / 700)
```

The key property: equal distances on the mel scale correspond to equal perceived pitch intervals.

```
Linear Hz:    100  200  400  800  1600  3200  6400  Hz
Mel:          150  283  496  835  1233  1646  2172  Mel
Mel spacing:   133  213  339  398   413   526      (increasing gaps in Hz)
```

Lower frequencies are more densely sampled (finer resolution where humans are sensitive). Higher frequencies are sparsely sampled (coarser resolution where humans are less sensitive). This is exactly what a TTS model needs to learn.

### 5.3 Mel filterbank

A **mel filterbank** is a set of triangular filters placed at mel-spaced center frequencies. Each filter is applied to the FFT magnitude spectrum to produce one mel bin:

```
FFT bins (513):  |---linear frequency 0 to 11025 Hz---|

Mel filter 1:         /\
Mel filter 2:            /\
Mel filter 3:               /\
...
Mel filter 80:                                  /\

Mel spectrum (80):  [b1  b2  b3  ...  b80]
```

Each mel bin `b_k` is the dot product of the FFT magnitude spectrum with filter `k`. The result is 80 values per time frame instead of 513.

**Why 80 mel bins?** `n_mels = 80` ([params.yaml:11](../params.yaml#L11)) is a standard value for TTS. It gives enough frequency resolution to distinguish phonemes while keeping the spectrogram compact. Common alternatives: 128 (higher quality, larger model input) or 40 (smaller, speech recognition).

### 5.4 Frequency bounds: `f_min` and `f_max`

```python
# params.yaml:12-13
f_min: 0.0
f_max: 8000.0
```

The mel filterbank only covers `[f_min, f_max]`. FFT energy outside this range is ignored.

`f_max = 8000.0` is chosen because:
1. Speech energy is concentrated below 8 kHz
2. Vietnamese consonants and tones are below 8 kHz  
3. A Nyquist of 11025 Hz gives comfortable headroom above 8000 Hz

### 5.5 Mel spectrogram in code

```python
# feature_extractor.py:30-43
mel_transform = T.MelSpectrogram(
    sample_rate=22050,
    n_fft=1024,
    hop_length=256,
    win_length=1024,
    n_mels=80,
    f_min=0.0,
    f_max=8000.0,
    power=1.0,       # magnitude (not power) spectrogram as STFT input
    center=True,
    pad_mode="reflect",
)
mel = mel_transform(waveform)   # shape: (1, 80, T)
```

`power=1.0` means the STFT is computed as **magnitude** (`|X|`), not power (`|X|²`). For TTS, magnitude spectrograms are more common because HiFi-GAN was trained with them.

---

## Module 6 — Log-Mel Spectrogram

### 6.1 Why take the log

A raw mel spectrogram has values that span several orders of magnitude. Loud sounds have energy 10,000× stronger than quiet sounds. This extreme range causes problems for neural networks:

1. **Gradient issues:** large values dominate gradients during training
2. **Perceptual mismatch:** humans perceive loudness logarithmically (the decibel scale is logarithmic)
3. **Compression:** log maps 4 orders of magnitude to a manageable linear range

### 6.2 `AmplitudeToDB`

```python
# feature_extractor.py:42
mel_db = T.AmplitudeToDB(stype="magnitude", top_db=80)(mel)
```

This converts linear mel amplitude to decibels:

```
dB = 20 * log10(amplitude / max_amplitude)    [for magnitude, stype="magnitude"]
dB = 10 * log10(power / max_power)            [for stype="power"]
```

`top_db=80` clamps the minimum value: nothing goes below `(max_dB - 80)`. This prevents `-inf` from appearing when a mel bin has zero energy (silence).

**Concrete example:**
```
mel amplitude values before log:  [0.001,  0.1,   1.0,  10.0]
After AmplitudeToDB (top_db=80):  [-60dB, -20dB,  0dB, +20dB]
After clamping at top_db=80:      [-60dB, -20dB,  0dB, +20dB]  ← all within 80dB range
```

### 6.3 The final tensor shape

After all transforms, a single audio file becomes:

```python
mel_db.squeeze(0).numpy()   # shape: (80, T)
```

- **Rows** (80): mel frequency bins, low frequency at top (bin 0) to high frequency at bottom (bin 79)
- **Columns** (T): time frames, T = audio_length_in_samples / hop_length

For a 3-second clip at 22050 Hz with hop_length=256:
```
T = (3 × 22050) / 256 ≈ 258 frames
Final shape: (80, 258)
```

This `(80, T)` matrix is what the Matcha-TTS acoustic model receives as its training target.

### 6.4 Normalization of mel statistics

After extracting all training mels, the pipeline computes global stats:

```python
# feature_extractor.py:128-137
def compute_stats(mels: dict[str, np.ndarray], out_path: Path) -> dict[str, float]:
    all_mels = np.concatenate([m.reshape(m.shape[0], -1) for m in mels.values()], axis=1)
    stats = {
        "mel_mean": float(all_mels.mean()),
        "mel_std": float(all_mels.std()),
        ...
    }
```

These stats are used to z-score normalize the mel spectrograms before passing them to the model:
```
normalized = (mel - mel_mean) / mel_std
```

This ensures the neural network input has zero mean and unit variance, which stabilizes training.

---

## Module 7 — Peak Normalization and Dynamic Range

### 7.1 The decibel scale (dB)

The decibel is a logarithmic unit for comparing two values. For amplitude:
```
dB = 20 * log10(amplitude / reference)
```

**dBFS (decibels relative to Full Scale):** 0 dBFS is the maximum possible digital amplitude (1.0 in floating-point). Everything is at or below 0 dBFS.

```
amplitude = 1.0   →  0 dBFS
amplitude = 0.5   →  −6 dBFS
amplitude = 0.1   →  −20 dBFS
amplitude = 0.01  →  −40 dBFS
```

### 7.2 Peak normalization

**Peak normalization** sets the loudest sample in the recording to a target level (in dBFS) and scales all other samples proportionally:

```python
# audio_processor.py:41-44
peak = waveform.abs().max()
if peak > 0:
    target_linear = 10 ** (peak_normalize_db / 20)  # dBFS → linear amplitude
    waveform = waveform * (target_linear / peak)
```

With `peak_normalize_db = -3.0`:
```
target_linear = 10^(-3/20) = 10^(-0.15) ≈ 0.708
```

Every waveform's loudest sample becomes 0.708 (−3 dBFS).

**Why −3 dBFS, not 0 dBFS?**
- 0 dBFS (amplitude=1.0) leaves no headroom; any subsequent processing (e.g., resampling filter ringing) can clip
- −3 dBFS is a broadcast/mastering standard — enough headroom to be safe
- Consistent loudness across all training samples means the model doesn't learn loudness-dependent features

### 7.3 Peak vs RMS normalization

**Peak normalization** (what this project uses): scales so the loudest single sample hits the target. Fast transients (a loud /p/ burst) dominate.

**RMS normalization**: scales so the average energy hits the target. More representative of perceived loudness, but a loud spike can exceed 0 dBFS.

For TTS training data, peak normalization is safer — no clipping risk.

### 7.4 Mono conversion

```python
# audio_processor.py:27-28
if waveform.shape[0] > 1:
    waveform = waveform.mean(dim=0, keepdim=True)
```

Stereo audio has 2 channels. This project averages them to mono. TTS models are mono because:
1. Speech intelligibility is the same in mono
2. Halves the data size
3. Stereo models require 2× output synthesis channels — not needed here

---

## Module 8 — Signal-to-Noise Ratio (SNR)

### 8.1 What SNR measures

SNR is the ratio of speech energy to background noise energy, in decibels:

```
SNR_dB = 10 * log10(signal_power / noise_power)
```

| SNR | Audio quality |
|---|---|
| > 40 dB | Clean studio recording |
| 25–40 dB | Good home recording |
| 20–25 dB | Acceptable |
| 10–20 dB | Noisy — background sounds audible |
| < 10 dB | Very noisy — speech hard to understand |

This project rejects samples below 20 dB SNR ([params.yaml:18](../params.yaml#L18)). Grade A requires ≥ 35 dB ([validator.py:19](../src/tts_pipeline/data/validator.py#L19)).

### 8.2 The estimation algorithm used in this project

True SNR requires knowing the noise floor, which requires either a noise reference signal or a more complex algorithm (like WADA-SNR). This project uses a simpler heuristic:

```python
# audio_utils.py:37-44
def compute_snr_db(waveform: np.ndarray) -> float:
    signal_power = np.mean(waveform**2)          # overall RMS power
    if signal_power < 1e-10:
        return -np.inf

    frame_size = 512
    n_frames = len(waveform) // frame_size
    if n_frames < 2:
        return float("nan")

    frames = waveform[: n_frames * frame_size].reshape(n_frames, frame_size)
    frame_energies = np.mean(frames**2, axis=1)  # RMS energy per frame
    noise_power = float(np.percentile(frame_energies, 10))  # bottom 10% = noise estimate

    if noise_power < 1e-10:
        return float("inf")
    return float(10 * np.log10(signal_power / noise_power))
```

**The key assumption:** the quietest 10% of frames in a speech recording are likely to be background noise or near-silence between words. Their energy gives an estimate of the noise floor.

Step by step for a 2-second clip at 22050 Hz:
```
1. total samples = 44100
2. n_frames = 44100 // 512 = 86 frames
3. frame_energies = mean(frame²) for each of 86 frames → array of 86 values
4. noise_power = 10th percentile of those 86 values  (≈ 8–9 frames)
5. signal_power = mean(waveform²) across all samples
6. SNR = 10 * log10(signal_power / noise_power)
```

**Why this is a heuristic:** if the recording has continuous noise with no pauses, the "quietest 10% of frames" is still noisy — the estimate is too optimistic. But for typical TTS training data (read speech with pauses), it works well.

### 8.3 Frame size choice: 512 samples

```
512 samples / 22050 Hz ≈ 23 ms per frame
```

23 ms is a standard speech analysis frame length — long enough to capture multiple pitch cycles, short enough to treat as stationary. This same 23 ms frame length appears in STFT analysis (hop_length=256 ≈ 11.6 ms is even shorter).

---

## Module 9 — Voice Activity Detection (VAD)

### 9.1 What VAD does

VAD (Voice Activity Detection) automatically detects where speech starts and ends in a recording. Its job in this pipeline is to remove silence at the beginning and end of each audio file.

Why this matters:
- Recordings often start 0.5–2 seconds before speech (button press delay)
- Recordings often end with 0.5–1 second of trailing silence
- Training on silence teaches the model nothing about speech
- Mel spectrograms of silence are near-uniform low-energy frames that waste sequence length

### 9.2 The double-pass trick

```python
# audio_processor.py:35-38
if trim_silence:
    waveform = F.vad(waveform, sample_rate=target_sr)
    # Reverse + trim leading silence from end
    waveform = F.vad(waveform.flip(-1), sample_rate=target_sr).flip(-1)
```

`torchaudio.functional.vad()` trims **leading** silence (beginning of the file). To trim trailing silence, the signal is reversed, VAD is applied again (trimming what was the trailing silence, now at the beginning), and then the result is reversed back.

This two-line pattern elegantly solves both ends with one function call.

```
Original:  [silence][speech][silence]

Pass 1 (forward VAD):  [speech][silence]    ← leading silence removed
Reverse:               [silence][hceeps]

Pass 2 (VAD on reversed):  [hceeps]         ← trailing silence (now leading) removed
Reverse back:              [speech]
```

### 9.3 What `torchaudio.functional.vad` does internally

It uses a **statistical model** of speech energy. Specifically:
1. Computes short-time energy of the signal in overlapping frames
2. Models energy as a mixture of "speech" and "noise" distributions
3. Classifies each frame as speech or non-speech
4. Trims from the beginning until it finds the first speech frame

The key assumption: silence frames have consistently lower energy than speech frames. This works for most TTS training data.

### 9.4 VAD edge cases

- **Breath/lip smack at start:** May be classified as speech. Inspect edge cases if training quality is poor.
- **Whispering:** Energy is low — VAD may mistakenly trim it. Use low-energy mode or adjust thresholds if training on whispered speech.
- **Room tone / AC hum:** A constant low-frequency hum can confuse energy-based VAD. This is why the pipeline also checks SNR — VAD success doesn't guarantee clean audio.

### 9.5 Connecting VAD to SNR

VAD and SNR work together in this pipeline:

1. **VAD** removes silence frames → mel spectrogram only contains speech
2. **SNR check** in validator confirms the retained speech is above the noise floor

A file can pass VAD (speech was detected) but still fail SNR (the detected speech was too noisy). Both checks are needed.

---

## Module 10 — The Full Processing Pipeline, End to End

Now that you understand each piece, here is how they connect in this project:

```
Raw audio file (any sample rate, any channels)
        │
        ▼
[audio_processor.py: normalize_audio()]
  1. Load via torchaudio.load()          → waveform tensor (channels, samples)
  2. Mono conversion (mean channels)     → waveform (1, samples)
  3. Resample to 22050 Hz               → waveform (1, resampled_samples)
  4. VAD (double-pass)                  → waveform (1, trimmed_samples)
  5. Peak normalize to −3 dBFS          → waveform (1, trimmed_samples) in [−0.708, +0.708]
  6. Save as 16-bit WAV
        │
        ▼
[feature_extractor.py: extract_mel()]
  7. MelSpectrogram transform:
     a. STFT with n_fft=1024, hop=256, Hann window
     b. Mel filterbank (80 bins, 0–8000 Hz)
     c. AmplitudeToDB (top_db=80)        → mel tensor (1, 80, T)
  8. Squeeze and convert to numpy        → array (80, T)
  9. Save as .npy file
        │
        ▼
[feature_extractor.py: extract_f0()]
 10. CREPE or PYIN fundamental frequency extraction
     (same hop_length=256 for alignment)  → array (T,)
 11. Save as .npy file
        │
        ▼
[validator.py: validate_dataset()]
 12. Duration check (0.5s – 15.0s)
 13. SNR check (> 20 dB)
 14. Grade scorecard (A–F)
```

Every parameter decision in steps 1–14 is traceable to the `DataConfig` in [config.py](../src/tts_pipeline/utils/config.py) and [params.yaml](../params.yaml).

---

## Exercises

**Exercise 1** — Open a Python shell, load any `.wav` file with `torchaudio.load()`, and verify that:
- `waveform.shape[0]` is 1 (mono) or 2 (stereo)
- `waveform.abs().max()` is close to (but ≤) 1.0

**Exercise 2** — Given `sample_rate=22050`, `n_fft=1024`, `hop_length=256`, calculate:
- The frequency of bin 100
- The time (in seconds) of frame 50
- The number of mel frames for a 5-second clip
- The time resolution in milliseconds (one frame = how many ms?)

Answers:
```
bin_100_freq = 100 * (22050 / 1024) = 2153 Hz
frame_50_time = 50 * (256 / 22050) = 0.58 seconds
n_frames_5s = (5 * 22050) / 256 = 430 frames
time_resolution = 256 / 22050 * 1000 = 11.6 ms per frame
```

**Exercise 3** — Read `compute_snr_db` in [audio_utils.py:37-44](../src/tts_pipeline/utils/audio_utils.py#L37-L44). Create a synthetic waveform where you know the true SNR and verify the function's estimate is close:

```python
import numpy as np
from tts_pipeline.utils.audio_utils import compute_snr_db

sr = 22050
t = np.linspace(0, 1.0, sr, dtype=np.float32)

# True signal: 440 Hz sine at amplitude 0.5
signal = 0.5 * np.sin(2 * np.pi * 440 * t)

# Add white noise at known power
noise_amplitude = 0.05
noise = noise_amplitude * np.random.randn(sr).astype(np.float32)

# True SNR
true_snr = 10 * np.log10((0.5**2 / 2) / (noise_amplitude**2))
print(f"True SNR: {true_snr:.1f} dB")

# Estimated SNR
estimated_snr = compute_snr_db(signal + noise)
print(f"Estimated SNR: {estimated_snr:.1f} dB")
```

**Exercise 4** — Modify the mel extraction call in `feature_extractor.py` to use `n_mels=128` and `f_max=11025.0`. What does the output shape become for the same 3-second clip? Why might you use 128 bins instead of 80?

**Exercise 5** — Explain in plain language why `waveform.flip(-1)` achieves the same result as reversing the signal. What does `-1` refer to? What would `flip(0)` do instead?

---

## Mini Practice Project: `spectraview` — Audio Quality Inspector

This is a self-contained project you build from scratch. It has nothing to do with TTS. It practices every DSP concept from this course in isolation.

**What it does:** `spectraview` is a CLI tool that takes an audio file (or a directory of audio files), computes its DSP properties (SNR, duration, sample rate, peak level, mel frame count), and prints an ASCII spectrogram + quality report.

**What it practices:**

| Course concept | Where you use it in `spectraview` |
|---|---|
| Sampling rate, Nyquist | Report whether sample rate is sufficient for speech |
| STFT parameters | User-configurable `--n-fft`, `--hop-length` |
| Mel spectrogram | ASCII visualization of the mel spectrogram |
| Log-mel / AmplitudeToDB | The spectrogram values you display |
| Peak normalization | Report peak level in dBFS |
| SNR estimation | Quality grade for each file |
| VAD | Report trimmed vs untrimmed duration |

**Time estimate:** 3–5 hours.

---

### Step 0 — Project structure

```
spectraview/
├── pyproject.toml
├── src/
│   └── spectraview/
│       ├── __init__.py
│       ├── dsp.py          # core DSP functions
│       ├── visualizer.py   # ASCII spectrogram renderer
│       ├── report.py       # quality report builder
│       └── cli.py          # entry point
└── sample_audio/           # put .wav files here for testing
```

---

### Step 1 — `pyproject.toml`

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.backends.legacy:build"

[project]
name = "spectraview"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "torch>=2.3.0",
    "torchaudio>=2.3.0",
    "numpy>=1.26.0",
]

[project.optional-dependencies]
dev = ["mypy>=1.10.0", "pytest>=8.2.0"]

[project.scripts]
spectraview = "spectraview.cli:main"

[tool.setuptools.packages.find]
where = ["src"]

[tool.mypy]
python_version = "3.11"
strict = true
ignore_missing_imports = true
```

```bash
pip install -e ".[dev]"
```

---

### Step 2 — `dsp.py` (core signal processing)

Build these functions yourself. Use the formulas from the course. Check your answers against the implementations in the TTS project.

```python
from __future__ import annotations

import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
import torchaudio.functional as F
from pathlib import Path


def load(path: Path) -> tuple[np.ndarray, int]:
    """Load audio as mono float32 numpy array. Returns (waveform, sample_rate)."""
    waveform, sr = torchaudio.load(str(path))
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform.squeeze(0).numpy(), sr


def resample(waveform: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample waveform to target_sr."""
    if orig_sr == target_sr:
        return waveform
    tensor = torch.from_numpy(waveform).unsqueeze(0)
    resampled = torchaudio.transforms.Resample(orig_sr, target_sr)(tensor)
    return resampled.squeeze(0).numpy()


def peak_level_dbfs(waveform: np.ndarray) -> float:
    """Return peak amplitude in dBFS. 0 dBFS = amplitude 1.0."""
    peak = float(np.abs(waveform).max())
    if peak < 1e-10:
        return float("-inf")
    return 20 * np.log10(peak)


def peak_normalize(waveform: np.ndarray, target_dbfs: float = -3.0) -> np.ndarray:
    """Scale waveform so peak amplitude equals target_dbfs."""
    peak = float(np.abs(waveform).max())
    if peak < 1e-10:
        return waveform
    target_linear = 10 ** (target_dbfs / 20)
    return waveform * (target_linear / peak)


def compute_snr_db(waveform: np.ndarray, frame_size: int = 512) -> float:
    """
    Estimate SNR using 10th percentile frame energy as noise floor.
    Returns dB value, or nan/inf for edge cases.
    """
    signal_power = float(np.mean(waveform ** 2))
    if signal_power < 1e-10:
        return float("-inf")
    n_frames = len(waveform) // frame_size
    if n_frames < 2:
        return float("nan")
    frames = waveform[: n_frames * frame_size].reshape(n_frames, frame_size)
    frame_energies = np.mean(frames ** 2, axis=1)
    noise_power = float(np.percentile(frame_energies, 10))
    if noise_power < 1e-10:
        return float("inf")
    return float(10 * np.log10(signal_power / noise_power))


def trim_silence(waveform: np.ndarray, sample_rate: int) -> np.ndarray:
    """Trim leading and trailing silence using double-pass VAD."""
    tensor = torch.from_numpy(waveform).unsqueeze(0)
    trimmed = F.vad(tensor, sample_rate=sample_rate)
    trimmed = F.vad(trimmed.flip(-1), sample_rate=sample_rate).flip(-1)
    return trimmed.squeeze(0).numpy()


def mel_spectrogram(
    waveform: np.ndarray,
    sample_rate: int,
    n_fft: int = 1024,
    hop_length: int = 256,
    win_length: int = 1024,
    n_mels: int = 80,
    f_min: float = 0.0,
    f_max: float = 8000.0,
) -> np.ndarray:
    """Compute log-mel spectrogram. Returns array of shape (n_mels, T)."""
    tensor = torch.from_numpy(waveform).unsqueeze(0)
    transform = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels,
        f_min=f_min,
        f_max=f_max,
        power=1.0,
        center=True,
        pad_mode="reflect",
    )
    mel = transform(tensor)                           # (1, n_mels, T)
    mel_db = T.AmplitudeToDB(stype="magnitude", top_db=80)(mel)
    return mel_db.squeeze(0).numpy()                  # (n_mels, T)


def n_frames(n_samples: int, hop_length: int) -> int:
    """Calculate number of STFT frames."""
    return n_samples // hop_length


def nyquist_hz(sample_rate: int) -> int:
    return sample_rate // 2


def hz_per_fft_bin(sample_rate: int, n_fft: int) -> float:
    return sample_rate / n_fft


def ms_per_frame(hop_length: int, sample_rate: int) -> float:
    return hop_length / sample_rate * 1000
```

---

### Step 3 — `visualizer.py` (ASCII spectrogram)

This is the most fun part. You will render the mel spectrogram as ASCII characters in the terminal.

```python
from __future__ import annotations

import numpy as np

# Dense-to-sparse character gradient: full block → empty
_CHARS = " ░▒▓█"


def render_mel_ascii(
    mel_db: np.ndarray,
    width: int = 80,
    height: int = 20,
) -> str:
    """
    Render a (n_mels, T) log-mel spectrogram as an ASCII art string.
    Low frequencies at the bottom, high at the top (matches spectrogram convention).
    """
    n_mels, n_time = mel_db.shape

    # Downsample time axis to fit terminal width
    time_indices = np.linspace(0, n_time - 1, width, dtype=int)
    mel_ds = mel_db[:, time_indices]   # (n_mels, width)

    # Downsample frequency axis to fit terminal height
    freq_indices = np.linspace(0, n_mels - 1, height, dtype=int)
    mel_ds = mel_ds[freq_indices, :]   # (height, width)

    # Flip: row 0 = highest frequency (top of terminal)
    mel_ds = mel_ds[::-1, :]

    # Normalize to [0, 1]
    vmin = mel_ds.min()
    vmax = mel_ds.max()
    if vmax > vmin:
        mel_norm = (mel_ds - vmin) / (vmax - vmin)
    else:
        mel_norm = np.zeros_like(mel_ds)

    # Map to characters
    char_idx = (mel_norm * (len(_CHARS) - 1)).astype(int)
    lines = ["".join(_CHARS[char_idx[row, col]] for col in range(width))
             for row in range(height)]
    return "\n".join(lines)
```

---

### Step 4 — `report.py` (quality analysis)

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from spectraview.dsp import (
    compute_snr_db,
    hz_per_fft_bin,
    load,
    mel_spectrogram,
    ms_per_frame,
    n_frames,
    nyquist_hz,
    peak_level_dbfs,
    trim_silence,
)


@dataclass
class AudioReport:
    path: Path
    sample_rate: int
    duration_s: float
    trimmed_duration_s: float
    peak_dbfs: float
    snr_db: float
    n_mel_frames: int
    nyquist_hz: int
    hz_per_bin: float
    ms_per_frame: float
    quality_grade: str
    mel: "np.ndarray"  # type: ignore[name-defined]

    def __str__(self) -> str:
        return (
            f"File:            {self.path.name}\n"
            f"Sample rate:     {self.sample_rate} Hz  (Nyquist: {self.nyquist_hz} Hz)\n"
            f"Duration:        {self.duration_s:.2f}s  →  {self.trimmed_duration_s:.2f}s after VAD\n"
            f"Peak level:      {self.peak_dbfs:.1f} dBFS\n"
            f"SNR:             {self.snr_db:.1f} dB\n"
            f"Mel frames:      {self.n_mel_frames}  ({self.ms_per_frame:.1f} ms/frame)\n"
            f"Hz per FFT bin:  {self.hz_per_bin:.1f} Hz\n"
            f"Quality grade:   {self.quality_grade}\n"
        )


def _grade(snr_db: float, duration_s: float) -> str:
    if duration_s < 0.3:
        return "F (too short)"
    if snr_db >= 35:
        return "A"
    if snr_db >= 30:
        return "B"
    if snr_db >= 25:
        return "C"
    if snr_db >= 20:
        return "D"
    return "F (noisy)"


def analyze(
    path: Path,
    n_fft: int = 1024,
    hop_length: int = 256,
    n_mels: int = 80,
    target_sr: int = 22050,
) -> AudioReport:
    import numpy as np
    from spectraview.dsp import resample

    waveform, sr = load(path)
    if sr != target_sr:
        waveform = resample(waveform, sr, target_sr)
        sr = target_sr

    duration = len(waveform) / sr
    trimmed = trim_silence(waveform, sr)
    trimmed_duration = len(trimmed) / sr

    return AudioReport(
        path=path,
        sample_rate=sr,
        duration_s=duration,
        trimmed_duration_s=trimmed_duration,
        peak_dbfs=peak_level_dbfs(waveform),
        snr_db=compute_snr_db(waveform),
        n_mel_frames=n_frames(len(waveform), hop_length),
        nyquist_hz=nyquist_hz(sr),
        hz_per_bin=hz_per_fft_bin(sr, n_fft),
        ms_per_frame=ms_per_frame(hop_length, sr),
        quality_grade=_grade(compute_snr_db(waveform), trimmed_duration),
        mel=mel_spectrogram(trimmed, sr, n_fft, hop_length, win_length=n_fft, n_mels=n_mels),
    )
```

---

### Step 5 — `cli.py` (entry point)

```python
from __future__ import annotations

import argparse
from pathlib import Path

from spectraview.report import analyze
from spectraview.visualizer import render_mel_ascii


def _process_file(path: Path, args: argparse.Namespace) -> None:
    print(f"\n{'='*80}")
    report = analyze(
        path,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        n_mels=args.n_mels,
        target_sr=args.sample_rate,
    )
    print(report)
    if not args.no_plot:
        print("Mel spectrogram (high freq top, low freq bottom):")
        print(render_mel_ascii(report.mel, width=80, height=16))
    print(f"{'='*80}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audio quality inspector")
    parser.add_argument("path", type=Path, help=".wav file or directory")
    parser.add_argument("--sample-rate", type=int, default=22050)
    parser.add_argument("--n-fft", type=int, default=1024)
    parser.add_argument("--hop-length", type=int, default=256)
    parser.add_argument("--n-mels", type=int, default=80)
    parser.add_argument("--no-plot", action="store_true", help="Skip ASCII spectrogram")
    args = parser.parse_args()

    target = args.path
    if target.is_dir():
        files = sorted(target.rglob("*.wav"))
        if not files:
            print(f"No .wav files found in {target}")
            return
        for f in files:
            _process_file(f, args)
    elif target.is_file():
        _process_file(target, args)
    else:
        print(f"Path not found: {target}")


if __name__ == "__main__":
    main()
```

---

### Step 6 — Generate sample audio and run it

Create `make_samples.py` at the project root:

```python
import numpy as np
import soundfile as sf
from pathlib import Path

sr = 22050
Path("sample_audio").mkdir(exist_ok=True)

# Clean sine wave (high SNR)
t = np.linspace(0, 2.0, 2 * sr, dtype=np.float32)
clean = 0.5 * np.sin(2 * np.pi * 440 * t)
sf.write("sample_audio/clean_440hz.wav", clean, sr)

# Noisy speech simulation (low SNR)
speech = 0.3 * np.sin(2 * np.pi * 200 * t) + 0.15 * np.sin(2 * np.pi * 400 * t)
noise = 0.1 * np.random.randn(len(t)).astype(np.float32)
sf.write("sample_audio/noisy_speech.wav", speech + noise, sr)

# With leading and trailing silence
silence = np.zeros(sr, dtype=np.float32)  # 1 second silence
padded = np.concatenate([silence, clean, silence])
sf.write("sample_audio/padded_silence.wav", padded, sr)

print("Created 3 sample audio files.")
```

```bash
python make_samples.py
spectraview sample_audio/
```

**Expected output** for `clean_440hz.wav`:
```
================================================================================
File:            clean_440hz.wav
Sample rate:     22050 Hz  (Nyquist: 11025 Hz)
Duration:        2.00s  →  2.00s after VAD
Peak level:      -6.0 dBFS
SNR:             XX.X dB
Mel frames:      172  (11.6 ms/frame)
Hz per FFT bin:  21.5 Hz
Quality grade:   A

Mel spectrogram (high freq top, low freq bottom):
                                                                                
                                                                                
...
████████████████████████████████████████████████████████████████████████████████
...
================================================================================
```

For `padded_silence.wav`, the duration after VAD should be shorter than 3 seconds (the 2 seconds of silence trimmed off).

---

### Step 7 — Experiment: change the parameters

Run these one by one and observe the output. Before running each, predict what will change:

```bash
# Halve frequency resolution, double time resolution
spectraview sample_audio/clean_440hz.wav --n-fft 512

# Use more mel bins (finer frequency detail)
spectraview sample_audio/clean_440hz.wav --n-mels 128

# Wider time steps (fewer frames per second)
spectraview sample_audio/clean_440hz.wav --hop-length 512

# Target a different sample rate (will trigger resampling)
spectraview sample_audio/clean_440hz.wav --sample-rate 16000
```

For each run, verify your prediction against the output. The key numbers to watch:
- `Hz per FFT bin` changes when you change `--n-fft`
- `ms/frame` changes when you change `--hop-length`
- `Mel frames` changes when you change either
- The ASCII spectrogram shape should reflect the frequency content you expect

---

### Step 8 — Add a batch SNR histogram

Add a `--histogram` flag that, when analyzing a directory, prints an ASCII histogram of SNR values:

```
SNR Distribution:
  35+ dB  ████████████████  (8 files)  Grade A
  30-35   ████████          (4 files)  Grade B
  25-30   ████              (2 files)  Grade C
  20-25   ██                (1 file)   Grade D
  <20     █                 (1 file)   Grade F
```

This is a miniature version of `validator.py`'s `snr_distribution.json`. The implementation requires:
- Collecting `AudioReport` objects for all files
- Binning SNR values
- Rendering the histogram with fixed-width bars

---

### Completion Checklist

- [ ] Explain why 22050 Hz captures human speech but 8000 Hz does not fully
- [ ] Given `n_fft` and `sample_rate`, calculate the frequency of any FFT bin without looking at notes
- [ ] Explain in one sentence what `hop_length` controls and what changing it trades off
- [ ] Explain why `AmplitudeToDB` uses `top_db=80` and what would happen without it
- [ ] Run `spectraview` on a silent file and explain the output
- [ ] Explain the double-pass VAD trick (`flip(-1)`) to someone who has not seen it
- [ ] Synthesize a tone at a specific frequency (e.g., 880 Hz) and verify it appears in the correct mel bin of the ASCII spectrogram
- [ ] Explain why the TTS project uses `power=1.0` (magnitude) rather than `power=2.0` (power spectrum)