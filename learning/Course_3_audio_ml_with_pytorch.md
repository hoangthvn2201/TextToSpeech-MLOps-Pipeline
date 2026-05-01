# Course 3: Audio ML with PyTorch (`torchaudio` + `librosa`)

**Prerequisites:** Course 1 (Python for ML Engineers), Course 2 (Digital Signal Processing for Audio).
**Goal:** After this course you can read every audio I/O and feature extraction file in this project — understanding not just *what* the code does but *why* each library is chosen and *what shape* data is at every step.

---

## Module 1 — Audio File Formats: What Lives on Disk

Before you call `torchaudio.load()`, you need to know what the file actually contains.

### 1.1 Container vs codec

Every audio file has two layers:

| Layer | What it is | Examples |
|---|---|---|
| **Container** | Wraps data with metadata (sample rate, channels, bit depth) | WAV, FLAC, OGG |
| **Codec** | How the audio data inside is encoded/compressed | PCM (uncompressed), Vorbis, MP3 |

A `.wav` file is a **RIFF container** holding **PCM** (Pulse Code Modulation) data — raw, uncompressed samples. No compression math involved when reading it. This is why the project saves normalized audio as WAV ([audio_processor.py:47](../src/tts_pipeline/data/audio_processor.py#L47)).

### 1.2 Why WAV for training data

The ingestion stage accepts `.wav`, `.flac`, `.mp3`, `.ogg` ([ingestion.py:7](../src/tts_pipeline/data/ingestion.py#L7)):

```python
_AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg"}
```

But after normalization it always writes WAV. The reasons:

| Property | WAV/PCM | MP3/OGG |
|---|---|---|
| Lossless | Yes | No |
| Random access (seek to sample N) | O(1) | O(n) — must decode from start |
| Decode speed | Fast (memcpy) | Slow (DSP decode) |
| File size | Large | Small |

Training iterates over files thousands of times. Random access and decode speed dominate; storage is cheap.

### 1.3 WAV file anatomy

```
RIFF header (44 bytes):
  "RIFF" magic            4 bytes
  file size               4 bytes (little-endian uint32)
  "WAVE" format           4 bytes
  "fmt " chunk marker     4 bytes
  chunk size = 16         4 bytes
  audio format = 1 (PCM)  2 bytes
  num channels            2 bytes
  sample rate             4 bytes
  byte rate               4 bytes
  block align             2 bytes
  bits per sample         2 bytes
  "data" chunk marker     4 bytes
  data size               4 bytes

PCM data (N bytes):
  sample_0  sample_1  sample_2  ...  (16-bit signed integers for this project)
```

This exact structure is manually constructed in [audio_utils.py:47-56](../src/tts_pipeline/utils/audio_utils.py#L47-L56) for the streaming endpoint — the WAV header is emitted first so the client can start playing before the full audio is generated.

---

## Module 2 — `soundfile`: The Workhorse for Audio I/O

`soundfile` (backed by libsndfile) is the most reliable pure-Python library for reading and writing audio files. It is used throughout this project for cases where you need a numpy array directly.

### 2.1 `sf.read` — load audio as numpy

```python
import soundfile as sf

data, sr = sf.read("speech.wav")
# data: numpy array, float64 by default, shape (n_samples,) for mono or (n_samples, channels) for stereo
# sr:   integer sample rate
```

Key parameter: `dtype`

```python
data, sr = sf.read("speech.wav", dtype="float32")   # float32 for torch/numpy compat
data, sr = sf.read("speech.wav", dtype="int16")      # raw PCM integers
```

**In this project:** [validator.py:63](../src/tts_pipeline/data/validator.py#L63) reads audio specifically to compute SNR:
```python
data, _ = sf.read(str(audio_path))
snr = compute_snr_db(data.astype(np.float32))
```

And in tests ([test_audio_processor.py:13](../tests/unit/test_audio_processor.py#L13)):
```python
data, sr = sf.read(str(dst))
assert sr == 16000
```

### 2.2 `sf.write` — save audio from numpy

```python
import numpy as np
import soundfile as sf

waveform = np.zeros(22050, dtype=np.float32)  # 1 second of silence
sf.write("output.wav", waveform, 22050)        # infers PCM_16 from float32 input

# Explicit subtype control:
sf.write("output.wav", waveform, 22050, subtype="PCM_24")
```

**In this project:** [audio_utils.py:24](../src/tts_pipeline/utils/audio_utils.py#L24) uses `sf.write` for general audio saving:
```python
sf.write(str(path), waveform_np, sr)
```

### 2.3 `sf.info` — metadata without loading data

Reading a 500 MB audio file just to check its sample rate is wasteful. `sf.info` reads only the header:

```python
info = sf.info("speech.wav")
print(info.samplerate)   # 22050
print(info.channels)     # 1
print(info.frames)       # total samples
print(info.duration)     # seconds = frames / samplerate
print(info.format)       # 'WAV'
print(info.subtype)      # 'PCM_16'
```

**In this project:** [ingestion.py:46-47](../src/tts_pipeline/data/ingestion.py#L46-L47):
```python
info = sf.info(str(audio_path))
duration = info.frames / info.samplerate
```

Duration is computed from metadata — no audio decode needed.

### 2.4 `sf.read` vs `torchaudio.load` — when to use which

| Scenario | Use |
|---|---|
| Need a numpy array | `sf.read` |
| Need a torch Tensor | `torchaudio.load` |
| Need only metadata (sr, duration) | `sf.info` |
| Resampling immediately after load | `torchaudio.load` (can chain with `Resample`) |
| Writing audio from numpy | `sf.write` |
| Writing audio from a Tensor | `torchaudio.save` |

---

## Module 3 — `torchaudio.load`: The Primary Tensor I/O

### 3.1 Return values and shape convention

```python
import torchaudio

waveform, sr = torchaudio.load("speech.wav")
# waveform: torch.Tensor, dtype=float32, shape=(channels, n_samples)
# sr:       int
```

**Critical shape: `(channels, n_samples)` — channels first.**

```python
waveform.shape   # torch.Size([1, 44100]) for mono 2-second clip at 22050 Hz
```

This is the `(C, T)` convention. It is the opposite of `soundfile` which returns `(T,)` for mono or `(T, C)` for stereo. Always check the shape explicitly when mixing the two libraries.

### 3.2 Amplitude range

`torchaudio.load` normalizes integer PCM to `[-1.0, 1.0]` float32 automatically. A 16-bit WAV sample of value `32767` becomes `1.0`. You never see raw PCM integers.

### 3.3 Mono conversion

```python
# audio_processor.py:27-28
if waveform.shape[0] > 1:
    waveform = waveform.mean(dim=0, keepdim=True)
```

`dim=0` is the channel dimension. `keepdim=True` preserves the `(1, T)` shape after averaging — without it you'd get `(T,)`, losing the channel dimension that downstream code expects.

### 3.4 `torchaudio.save` — writing Tensors to disk

```python
# audio_processor.py:47
torchaudio.save(str(dst), waveform, target_sr, encoding="PCM_S", bits_per_sample=16)
```

| Parameter | Value | Meaning |
|---|---|---|
| `encoding="PCM_S"` | Signed PCM | Standard WAV encoding |
| `bits_per_sample=16` | 16-bit | 65536 amplitude levels |

`waveform` must be a float32 Tensor of shape `(channels, samples)` — the same shape `torchaudio.load` returns.

### 3.5 `torchaudio.info` — metadata from Tensor path

```python
info = torchaudio.info("speech.wav")
print(info.sample_rate)       # 22050
print(info.num_channels)      # 1
print(info.num_frames)        # total samples
print(info.bits_per_sample)   # 16
print(info.encoding)          # PCM_S
```

Like `sf.info`, this reads only the header — useful for validation before loading.

---

## Module 4 — `torchaudio.transforms`: Stateful Transforms as `nn.Module`

### 4.1 The `nn.Module` pattern for audio transforms

`torchaudio.transforms` classes are subclasses of `torch.nn.Module`. This means:
- They have learnable parameters (or fixed ones)
- They are called like functions: `transform(input)`
- They can be moved to GPU: `transform.to("cuda")`
- They can be composed with `nn.Sequential`

```python
import torchaudio.transforms as T

mel_transform = T.MelSpectrogram(sample_rate=22050, n_fft=1024, n_mels=80)
mel = mel_transform(waveform)   # __call__ invokes forward()
```

**Why this matters:** you can build a `nn.Sequential` preprocessing pipeline that runs on GPU — the same graph PyTorch uses for model inference.

### 4.2 `Resample` — changing sample rate

```python
resampler = torchaudio.transforms.Resample(orig_freq=44100, new_freq=22050)
waveform_22k = resampler(waveform_44k)
```

Internally this:
1. Computes a low-pass filter at the new Nyquist (11025 Hz)
2. Applies the filter to avoid aliasing
3. Takes every other sample (decimation)

**In this project:** [audio_processor.py:32](../src/tts_pipeline/data/audio_processor.py#L32):
```python
waveform = torchaudio.transforms.Resample(sr, target_sr)(waveform)
```

A new `Resample` object is created per call here. In a hot loop you'd cache it:
```python
# More efficient when resampling many files with the same ratio:
resampler = torchaudio.transforms.Resample(sr, target_sr)
for waveform in waveforms:
    out = resampler(waveform)
```

### 4.3 `MelSpectrogram` — the full pipeline in one transform

```python
# feature_extractor.py:29-42
mel_transform = T.MelSpectrogram(
    sample_rate=22050,
    n_fft=1024,
    hop_length=256,
    win_length=1024,
    n_mels=80,
    f_min=0.0,
    f_max=8000.0,
    power=1.0,
    center=True,
    pad_mode="reflect",
)
mel = mel_transform(waveform)   # input: (1, T_samples), output: (1, 80, T_frames)
```

Under the hood, `MelSpectrogram` is:
```
waveform → STFT → |·| (magnitude) → MelFilterbank → mel spectrogram
```

All in one step. The internal `Spectrogram` transform does the STFT; a `MelScale` transform applies the filterbank.

**Parameter recap from Course 2:**

| Parameter | Value | Controls |
|---|---|---|
| `n_fft` | 1024 | FFT size → frequency resolution (21.5 Hz/bin) |
| `hop_length` | 256 | Step between frames → time resolution (11.6 ms/frame) |
| `win_length` | 1024 | Window width (= n_fft here, common choice) |
| `n_mels` | 80 | Number of mel filterbank bands |
| `power` | 1.0 | Magnitude spectrum (not power spectrum) |
| `center` | True | Pad signal so frame 0 is centered on sample 0 |
| `f_min/f_max` | 0/8000 | Frequency bounds of the filterbank |

### 4.4 `AmplitudeToDB` — linear amplitude → decibels

```python
# feature_extractor.py:42
mel_db = T.AmplitudeToDB(stype="magnitude", top_db=80)(mel)
```

```
out = 20 * log10(in / max(in))    [for stype="magnitude"]
out = clamp(out, min = max(out) - top_db)
```

`top_db=80` ensures no value is more than 80 dB below the maximum. Without this, a silent mel bin would produce `-inf` (log of 0), which breaks training.

### 4.5 Composing transforms with `nn.Sequential`

```python
# You could write the full pipeline as:
pipeline = torch.nn.Sequential(
    T.MelSpectrogram(sample_rate=22050, n_fft=1024, hop_length=256, n_mels=80),
    T.AmplitudeToDB(stype="magnitude", top_db=80),
)
mel_db = pipeline(waveform)   # (1, 80, T)
```

This project doesn't use `Sequential` (it uses separate calls for clarity), but the option exists. On GPU, the entire chain runs without CPU round-trips.

---

## Module 5 — `torchaudio.functional`: Stateless Operations

`torchaudio.functional` contains the same operations as `torchaudio.transforms` but as plain functions — no `__init__`, no state, no caching. Use them for one-off operations or when you already have the filter coefficients.

### 5.1 `vad` — Voice Activity Detection

```python
import torchaudio.functional as F

# audio_processor.py:36-38
waveform = F.vad(waveform, sample_rate=target_sr)
waveform = F.vad(waveform.flip(-1), sample_rate=target_sr).flip(-1)
```

`F.vad` trims **leading** silence from a waveform tensor. It does not need to be created as a transform object because it has no learnable state — just a statistical model of speech energy.

**Signature:**
```python
torchaudio.functional.vad(
    waveform: Tensor,           # (1, T) — must be mono
    sample_rate: int,
    trigger_level: float = 7.0, # sensitivity; higher = less aggressive
    ...
) -> Tensor                     # (1, T_trimmed) — same channel dim, shorter time
```

### 5.2 `transforms` vs `functional` — decision rule

| Use `transforms` when | Use `functional` when |
|---|---|
| Called many times (caches filter coefficients) | Called once per program run |
| You want GPU movement via `.to()` | State doesn't matter |
| You want `nn.Sequential` composition | You want a simple function call |

For `Resample` in a loop: use `transforms` (the sinc filter is precomputed once).
For `vad` on each file: `functional` is fine (no expensive precompute).

---

## Module 6 — Fundamental Frequency (F0) Extraction

### 6.1 What F0 is and why TTS needs it

The **fundamental frequency (F0)** is the lowest frequency component of a voiced sound — essentially the pitch. For speech:

- Voiced sounds (vowels, nasals): F0 is the vibration rate of the vocal cords (~80–300 Hz for adults)
- Unvoiced sounds (/s/, /t/, /k/): no vocal cord vibration, F0 = 0 (or marked as unvoiced)

TTS models use F0 in two ways:
1. **Training signal:** the model learns to predict the pitch contour of the target speaker
2. **Conditioning:** some models can be given a target F0 at inference to control prosody

The F0 contour has the same time axis as the mel spectrogram — one F0 value per frame at `hop_length=256`.

### 6.2 `torchcrepe` — neural pitch detection (primary)

CREPE (Convolutional Representation for Pitch Estimation) is a neural network trained to estimate F0. It's more robust than signal-processing methods on noisy audio.

```python
# feature_extractor.py:48-62
import torchcrepe

waveform, sr = torchaudio.load(str(audio_path))
f0, periodicity = torchcrepe.predict(
    waveform,              # (1, T) — mono tensor
    sr,                    # sample rate
    hop_length=256,        # same as mel hop_length for alignment
    fmin=50.0,             # minimum expected F0 (Hz)
    fmax=1100.0,           # maximum expected F0 (Hz)
    model="tiny",          # "tiny" | "full" — smaller model = faster
    return_periodicity=True,  # also return voiced/unvoiced confidence
    batch_size=512,        # frames processed per forward pass
    device="cpu",          # or "cuda"
)
# f0:          (1, T_frames) — F0 in Hz, 0 for unvoiced
# periodicity: (1, T_frames) — confidence in [0, 1]
return f0.squeeze(0).numpy()   # (T_frames,)
```

**Why `fmin=50` and `fmax=1100`?**
- Adult male voice: 80–180 Hz
- Adult female voice: 165–255 Hz
- Children: up to ~400 Hz
- Vietnamese (tonal language): tone contours can reach high pitches
- 50–1100 Hz gives comfortable margin for all cases

### 6.3 `librosa.pyin` — probabilistic YIN (fallback)

PYIN is a signal-processing algorithm (no neural network). It's the fallback when `torchcrepe` is not installed:

```python
# feature_extractor.py:64-75
import librosa

y, _ = librosa.load(str(audio_path), sr=sample_rate)
# librosa.load returns (numpy_array, sr) — note: no channel dim, just (T,)

f0, voiced_flag, voiced_probs = librosa.pyin(
    y,
    fmin=50.0,
    fmax=1100.0,
    sr=sample_rate,
    hop_length=256,
)
# f0:          (T_frames,) — F0 in Hz, NaN for unvoiced frames
# voiced_flag: (T_frames,) — boolean
# voiced_probs:(T_frames,) — probability

f0 = np.nan_to_num(f0, nan=0.0)    # replace NaN with 0.0 for unvoiced frames
return f0.astype(np.float32)
```

**Important difference from torchcrepe:** PYIN returns `NaN` for unvoiced frames. `np.nan_to_num(f0, nan=0.0)` converts them to 0, which is the convention in this project. Failure to do this would propagate NaN through all downstream computations.

### 6.4 `librosa.load` vs `torchaudio.load`

| | `librosa.load` | `torchaudio.load` |
|---|---|---|
| Return type | `(np.ndarray, int)` | `(torch.Tensor, int)` |
| Array shape | `(T,)` for mono | `(C, T)` — channels first |
| Resampling | Built-in via `sr=` param | Separate `Resample` transform |
| Default dtype | float32 | float32 |
| Default behavior | Resamples to 22050 Hz! | Loads at original sample rate |

The `librosa.load(path, sr=sample_rate)` call explicitly sets the target sample rate — librosa will resample on load if needed. This is convenient but hides the sample rate conversion.

### 6.5 F0 alignment with mel spectrogram

Both F0 and mel use `hop_length=256`. For a 1-second clip at 22050 Hz:

```
mel shape:  (80, 86)    — 86 time frames
f0 shape:   (86,)       — 86 F0 values

They are aligned: f0[t] corresponds to mel[:, t]
```

This alignment is essential. The duration predictor in Matcha-TTS needs to match phoneme durations to both mel frames and F0 values simultaneously.

### 6.6 `np.nan_to_num` — safe NaN handling

```python
f0 = np.nan_to_num(f0, nan=0.0)
```

`NaN` (Not a Number) propagates silently through numpy operations. `mean([1.0, NaN, 2.0])` gives `NaN`. One NaN in a batch of F0 values would zero out gradients for the entire batch. Always replace NaN before saving:

```python
import numpy as np

arr = np.array([1.0, np.nan, 3.0, np.nan])
np.nan_to_num(arr, nan=0.0)     # array([1., 0., 3., 0.])
np.nan_to_num(arr, nan=-1.0)    # array([ 1., -1.,  3., -1.])
np.nanmean(arr)                  # 2.0 — nanmean ignores NaN instead of replacing
```

---

## Module 7 — Audio → Bytes: WAV Format and Streaming

The serving layer needs to send audio over HTTP. You can't send a numpy array through JSON. You convert it to bytes in WAV format.

### 7.1 `audio_to_wav_bytes` — the complete conversion

```python
# audio_utils.py:26-30
def audio_to_wav_bytes(waveform: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)          # 16-bit = 2 bytes per sample
        wf.setframerate(sr)
        pcm = (waveform * 32767).clip(-32768, 32767).astype(np.int16)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()
```

Step by step:

| Step | Code | What happens |
|---|---|---|
| 1. Create in-memory buffer | `io.BytesIO()` | A file-like object in RAM, not on disk |
| 2. Open WAV writer | `wave.open(buf, "wb")` | Writes RIFF/WAV header to buffer |
| 3. Configure channels | `setnchannels(1)` | Mono |
| 4. Configure bit depth | `setsampwidth(2)` | 16-bit = 2 bytes per sample |
| 5. Configure sample rate | `setframerate(sr)` | 22050 Hz |
| 6. Convert float → int16 | `waveform * 32767` | Float32 [-1, 1] → Int16 [-32768, 32767] |
| 7. Clip to safe range | `.clip(-32768, 32767)` | Prevent overflow from peak values above 1.0 |
| 8. Write PCM bytes | `wf.writeframes(pcm.tobytes())` | Append raw PCM to WAV file |
| 9. Return bytes | `buf.getvalue()` | Complete WAV file as bytes |

### 7.2 Why `* 32767` (not `* 32768`)?

Int16 range: `[-32768, 32767]`. If the float is exactly `1.0`, then `1.0 * 32768 = 32768` — overflows Int16. Using `32767` ensures `1.0` maps to `32767`, which is safely inside the range. The 1-unit asymmetry of int16 is the reason.

### 7.3 `wav_header_bytes` — streaming without knowing total size

For the streaming endpoint, the total audio size is unknown upfront (sentences are synthesized one at a time). The WAV header includes a "data size" field, but for streaming, it's set to 0 as a placeholder:

```python
# audio_utils.py:38-56
def wav_header_bytes(sr: int, n_channels: int = 1, bit_depth: int = 16) -> bytes:
    byte_rate = sr * n_channels * bit_depth // 8
    block_align = n_channels * bit_depth // 8
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",  # little-endian format string
        b"RIFF", 0,             # 0 = placeholder file size
        b"WAVE", b"fmt ",
        16, 1,                  # PCM format chunk
        n_channels, sr, byte_rate, block_align, bit_depth,
        b"data", 0,             # 0 = placeholder data size
    )
    return header
```

The streaming generator yields this header first, then yields raw PCM bytes sentence by sentence ([tts_stream.py:32-35](../src/tts_pipeline/serving/routes/tts_stream.py#L32-L35)):

```python
yield wav_header_bytes(engine.sample_rate)   # client starts playing
for sentence in sentences:
    mel = engine.synthesize_mel(sentence)
    audio_chunk = engine.vocode(mel)
    pcm = (audio_chunk.flatten() * 32767).clip(-32768, 32767).astype(np.int16)
    yield pcm.tobytes()                       # client receives and plays
```

### 7.4 Base64 for JSON transport

WAV bytes can't be embedded in JSON directly (JSON is UTF-8 text). Base64 encodes binary as ASCII text:

```python
# tts.py:41
audio_bytes_b64=base64.b64encode(wav_bytes).decode()
```

A 22050-sample mono WAV (1 second) is 44100 bytes → 58800 Base64 characters. The caller decodes with `base64.b64decode(response.audio_bytes_b64)`.

---

## Module 8 — The numpy ↔ Tensor Boundary

### 8.1 Why this boundary exists

`torchaudio` works with Tensors. `librosa`, `soundfile`, `scipy`, `numpy` work with arrays. The pipeline constantly crosses this boundary. Knowing the exact conversion calls prevents shape bugs.

### 8.2 Tensor → numpy

```python
# Pattern used throughout feature_extractor.py and audio_utils.py:
mel_db.squeeze(0).numpy()          # (1, 80, T) → (80, T) numpy array

# When the tensor requires grad or is on GPU:
tensor.detach().cpu().numpy()      # detach from graph, move to CPU, convert

# For 1D audio waveform:
audio_out = self._vocoder.run(...)[0]  # ONNX returns numpy directly
audio = audio_out.squeeze()            # (1, 1, T) → (T,) — pure numpy
```

### 8.3 numpy → Tensor

```python
# For librosa output (numpy array, shape (T,)):
waveform_np = np.zeros(22050, dtype=np.float32)
tensor = torch.from_numpy(waveform_np).unsqueeze(0)   # (T,) → (1, T)

# For batched processing:
batch = torch.from_numpy(np.stack([arr1, arr2, arr3]))  # (3, T)
```

`torch.from_numpy` shares memory with the numpy array — modifying one modifies the other. If you need independence, call `.clone()` on the Tensor.

### 8.4 `np.save` / `np.load` — feature caching

The feature extractor saves mel and F0 arrays to disk as `.npy` files and skips recomputation if they exist:

```python
# feature_extractor.py:84-96
if mel_out.exists() and f0_out.exists():
    mel = np.load(mel_out)                # fast: binary array load
    return audio_path.stem, mel

mel = extract_mel(audio_path, **cfg)      # slow: full audio → FFT → mel
np.save(mel_out, mel)                     # cache for next run
```

`.npy` format: a binary file with a small header (dtype, shape) followed by raw array bytes. `np.save` / `np.load` is the fastest way to persist numpy arrays — much faster than JSON or CSV.

### 8.5 `np.concatenate` for global statistics

```python
# feature_extractor.py:125
all_mels = np.concatenate([m.reshape(m.shape[0], -1) for m in mels.values()], axis=1)
```

This takes a dict of `(80, T_i)` mel arrays, reshapes each to `(80, T_i)` (already that shape), and concatenates along `axis=1` (time axis) to get `(80, T_total)`. Then computes global mean and std across all frames in the entire training set.

`m.reshape(m.shape[0], -1)` is a safe pattern: `-1` means "infer this dimension". For a `(80, T)` array it's a no-op. If the array were `(80, T, extra_dim)`, it would flatten the last two dims.

---

## Module 9 — Testing Audio Code

### 9.1 The fixture pattern — synthetic audio

Real audio files are large, copyrighted, and variable. Tests use synthetic sine waves:

```python
# conftest.py:12-18
@pytest.fixture(scope="session")
def synthetic_audio_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    audio_dir = tmp_path_factory.mktemp("audio")
    n_samples = int(SR * DURATION_SEC)   # 22050 * 2.0 = 44100 samples
    for i in range(N_SAMPLES):
        freq = 220 + i * 10              # each file has a different pitch
        t = np.linspace(0, DURATION_SEC, n_samples, endpoint=False)
        wave = (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
        sf.write(str(audio_dir / f"{i:04d}.wav"), wave, SR)
    return audio_dir
```

Key design decisions:
- `scope="session"`: the fixture is created once per test session, not once per test. 50 audio files written once → shared by all tests in the session. Use `scope="session"` for expensive setup.
- `tmp_path_factory`: creates a temp directory that persists for the session (vs `tmp_path` which is per-test)
- Each file has a unique frequency: tests can verify the correct file was processed by checking the expected frequency

### 9.2 What makes audio tests hard

**Floating-point sensitivity:** audio operations involve FFT, log, division. Results are not exactly reproducible across CPU architectures or library versions.

**Good:** `np.testing.assert_array_equal` — for deterministic operations (loading same file twice)

**Good:** `np.testing.assert_allclose(a, b, rtol=1e-3)` — for operations that should be nearly equal

**Bad:** `assert mel1 == mel2` — Python `==` on numpy arrays returns an array, not a bool

```python
# test_feature_extractor.py:13-16
def test_mel_deterministic(synthetic_audio_dir: Path) -> None:
    src = synthetic_audio_dir / "0001.wav"
    mel1 = extract_mel(src, ...)
    mel2 = extract_mel(src, ...)
    np.testing.assert_array_equal(mel1, mel2)   # exact equality — correct here
```

### 9.3 Testing peak normalization

```python
# test_audio_processor.py:14
def test_normalize_audio_peak(synthetic_audio_dir: Path, tmp_path: Path) -> None:
    src = synthetic_audio_dir / "0001.wav"
    dst = tmp_path / "peak.wav"
    normalize_audio(src, dst, target_sr=22050, peak_normalize_db=-3.0, trim_silence=False)
    data, _ = sf.read(str(dst))
    peak_db = 20 * np.log10(np.abs(data).max() + 1e-10)
    assert abs(peak_db - (-3.0)) < 1.0, f"Peak was {peak_db:.2f} dB, expected ~-3 dB"
```

`+ 1e-10` prevents `log10(0)` if the audio is exactly silent. The tolerance is `< 1.0 dB` — audio processing has rounding from 16-bit quantization, so exact equality would be fragile.

### 9.4 Testing sample rate after resample

```python
# test_audio_processor.py:12
def test_normalize_audio_resamples(synthetic_audio_dir: Path, tmp_path: Path) -> None:
    normalize_audio(src, dst, target_sr=16000, ...)
    data, sr = sf.read(str(dst))
    assert sr == 16000
    assert len(data) > 0
```

Use `sf.read` (not `torchaudio.load`) in tests for simplicity — you get both the data and sr in a familiar numpy shape. Testing the actual sample rate from the file metadata is more reliable than trusting the in-memory Tensor.

---

## Module 10 — Audio Evaluation: PESQ and STOI

### 10.1 What these metrics measure

**PESQ (Perceptual Evaluation of Speech Quality):** ITU standard for measuring speech codec quality. Uses a perceptual model of the human auditory system. Returns a score in `[-0.5, 4.5]` (wideband mode).

**STOI (Short-Time Objective Intelligibility):** Measures how intelligible speech is. Returns a score in `[0, 1]`. 1.0 = perfectly intelligible.

These are used in the two-tier evaluation gate ([params.yaml:73](../params.yaml#L73)):
```yaml
absolute_floor:
  stoi_mean: 0.60    # minimum intelligibility
```

### 10.2 The PESQ sample rate constraint

```python
# evaluation/metrics/pesq_stoi.py:23-30
if sample_rate not in (8000, 16000):
    import torchaudio, torch
    resample = torchaudio.transforms.Resample(sample_rate, 16000)
    synth_16k = resample(torch.from_numpy(synth).unsqueeze(0)).squeeze().numpy()
    ref_16k   = resample(torch.from_numpy(ref).unsqueeze(0)).squeeze().numpy()
    pesq_wb = pesq(16000, ref_16k, synth_16k, "wb")
```

`pesq` (the library) only accepts 8000 Hz or 16000 Hz input. This project uses 22050 Hz internally, so it resamples **just for PESQ computation** — not for training. This is a common pattern: the internal sample rate and the evaluation tool's required sample rate can differ.

### 10.3 Shape alignment for evaluation

```python
min_len = min(len(synth), len(ref))
synth = synth[:min_len].astype(np.float32)
ref   = ref[:min_len].astype(np.float32)
```

Synthesized and reference audio may differ by a few samples due to resampling or inference noise. Truncate both to the minimum length before computing PESQ/STOI. Mismatched lengths cause a crash, not a wrong answer.

---

## Module 11 — Putting It All Together: Data Flow Map

Every file in the audio pipeline, with exact array shapes at each stage:

```
Raw WAV file (any sr, any channels)
        │
        │  torchaudio.load()
        ▼
waveform: Tensor (C, T_raw)     ← C=1 or 2, T_raw varies
        │
        │  .mean(dim=0, keepdim=True)   [if stereo]
        ▼
waveform: Tensor (1, T_raw)     ← always mono now
        │
        │  torchaudio.transforms.Resample(sr, 22050)
        ▼
waveform: Tensor (1, T_22k)     ← T_22k = duration * 22050
        │
        │  torchaudio.functional.vad() × 2
        ▼
waveform: Tensor (1, T_trimmed) ← silence removed
        │
        │  peak normalization
        ▼
waveform: Tensor (1, T_trimmed) ← peak = 0.708 (-3 dBFS)
        │
        │  torchaudio.save(..., encoding="PCM_S", bits_per_sample=16)
        ▼
normalized .wav file on disk
        │
        │  torchaudio.load()
        ▼
waveform: Tensor (1, T_trimmed)
        │
        │  T.MelSpectrogram(n_fft=1024, hop=256, n_mels=80)
        ▼
mel: Tensor (1, 80, T_frames)   ← T_frames = T_trimmed // 256
        │
        │  T.AmplitudeToDB(top_db=80)
        ▼
mel_db: Tensor (1, 80, T_frames) ← values in dB
        │
        │  .squeeze(0).numpy()
        ▼
mel_np: ndarray (80, T_frames)  ← saved as .npy
        │
        ├─── librosa.load() → pyin() OR torchcrepe.predict()
        ▼
f0_np: ndarray (T_frames,)      ← saved as .npy, aligned with mel columns
```

---

## Exercises

**Exercise 1** — Load a `.wav` file with both `torchaudio.load` and `sf.read`. Compare their shapes. Write a function `load_as_tensor(path: Path) -> torch.Tensor` that wraps `sf.read` and returns a `(1, T)` float32 Tensor.

**Exercise 2** — Write a function that takes a `(1, T)` Tensor at any sample rate and returns a `(80, T_frames)` numpy array. Internally use `torchaudio.transforms.Resample`, `T.MelSpectrogram`, and `T.AmplitudeToDB`. Match the project's exact parameters.

**Exercise 3** — The `audio_to_wav_bytes` function uses `wave.open` from the standard library. Rewrite it using only `struct.pack` and `io.BytesIO` — construct the 44-byte WAV header manually using the format from Module 1. Compare the output bytes with the original function.

**Exercise 4** — In `extract_f0`, unvoiced frames become `0.0` after `np.nan_to_num`. This is correct but loses information — you can't distinguish "unvoiced" from "very low voiced pitch." Propose an alternative representation and modify `extract_f0` to return a tuple `(f0_array, voiced_mask)`.

**Exercise 5** — Read [conftest.py](../tests/conftest.py). The sine wave frequency is `220 + i * 10`. Write a test that verifies file `0010.wav` has a dominant frequency of `320 Hz` by computing the FFT and finding the peak bin.

---

## Mini Practice Project: `wavekit` — Audio Feature Extraction CLI

This is a self-contained project you build from scratch. No TTS dependency. Its purpose is to use every library from this course in a cohesive workflow.

**What it does:** `wavekit` scans a directory of raw audio files (any sample rate, any format), normalizes them, extracts mel spectrograms and F0, and saves the results as `.npy` files alongside a JSON manifest.

**What it practices:**

| Course concept | Where in `wavekit` |
|---|---|
| `soundfile.info` | Fast metadata scan before loading |
| `torchaudio.load` | Primary audio loading |
| `torchaudio.transforms.Resample` | Normalize to 22050 Hz |
| `MelSpectrogram` + `AmplitudeToDB` | Feature extraction |
| `librosa.pyin` | F0 extraction |
| `np.nan_to_num` | Safe F0 handling |
| `np.save` / `np.load` | Feature caching |
| `sf.write` | Save normalized WAV |
| `audio_to_wav_bytes` pattern | Understand WAV byte structure |
| `scope="session"` fixtures | Testing with synthetic audio |

**Time estimate:** 4–6 hours.

---

### Step 0 — Project structure

```
wavekit/
├── pyproject.toml
├── src/
│   └── wavekit/
│       ├── __init__.py
│       ├── io.py           # audio loading and saving
│       ├── features.py     # mel + f0 extraction
│       ├── normalize.py    # resampling + peak normalization
│       ├── manifest.py     # JSON manifest builder
│       └── cli.py          # entry point
└── tests/
    ├── conftest.py
    └── test_features.py
```

---

### Step 1 — `pyproject.toml`

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.backends.legacy:build"

[project]
name = "wavekit"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "torch>=2.3.0",
    "torchaudio>=2.3.0",
    "numpy>=1.26.0",
    "soundfile>=0.12.1",
    "librosa>=0.10.2",
]

[project.optional-dependencies]
dev = ["pytest>=8.2.0", "mypy>=1.10.0"]

[project.scripts]
wavekit = "wavekit.cli:main"

[tool.setuptools.packages.find]
where = ["src"]

[tool.mypy]
python_version = "3.11"
strict = true
ignore_missing_imports = true
```

---

### Step 2 — `io.py`

```python
from __future__ import annotations

import io
import struct
import wave
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio


SUPPORTED_EXTENSIONS = {".wav", ".flac", ".ogg", ".mp3"}


def get_info(path: Path) -> tuple[int, int, float]:
    """Return (sample_rate, channels, duration_s) without loading audio data."""
    info = sf.info(str(path))
    return info.samplerate, info.channels, info.duration


def load_mono_tensor(path: Path) -> tuple[torch.Tensor, int]:
    """Load audio as a (1, T) float32 Tensor at its native sample rate."""
    waveform, sr = torchaudio.load(str(path))
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform, sr


def save_wav(path: Path, waveform: np.ndarray, sr: int) -> None:
    """Save a (T,) float32 numpy array as a 16-bit WAV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), waveform.astype(np.float32), sr, subtype="PCM_16")


def to_wav_bytes(waveform: np.ndarray, sr: int) -> bytes:
    """Convert a (T,) float32 array to WAV bytes (in-memory, no disk write)."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        pcm = (waveform * 32767).clip(-32768, 32767).astype(np.int16)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


def scan_dir(audio_dir: Path) -> list[Path]:
    """Find all supported audio files in a directory (recursive)."""
    return [
        p for p in audio_dir.rglob("*")
        if p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
```

---

### Step 3 — `normalize.py`

```python
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torchaudio
import torchaudio.functional as F


def resample(waveform: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
    """Resample a (1, T) Tensor to target_sr."""
    if orig_sr == target_sr:
        return waveform
    return torchaudio.transforms.Resample(orig_sr, target_sr)(waveform)


def peak_normalize(waveform: torch.Tensor, target_dbfs: float = -3.0) -> torch.Tensor:
    """Scale waveform peak to target_dbfs."""
    peak = waveform.abs().max()
    if peak < 1e-10:
        return waveform
    target_linear = 10 ** (target_dbfs / 20)
    return waveform * (target_linear / peak)


def trim_silence(waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """Remove leading and trailing silence using double-pass VAD."""
    trimmed = F.vad(waveform, sample_rate=sample_rate)
    trimmed = F.vad(trimmed.flip(-1), sample_rate=sample_rate).flip(-1)
    return trimmed


def process(
    waveform: torch.Tensor,
    orig_sr: int,
    target_sr: int = 22050,
    peak_dbfs: float = -3.0,
    do_trim: bool = True,
) -> torch.Tensor:
    """Full normalization pipeline: resample → VAD → peak normalize."""
    waveform = resample(waveform, orig_sr, target_sr)
    if do_trim:
        waveform = trim_silence(waveform, target_sr)
    waveform = peak_normalize(waveform, peak_dbfs)
    return waveform
```

---

### Step 4 — `features.py`

```python
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torchaudio.transforms as T


def extract_mel(
    waveform: torch.Tensor,
    sample_rate: int = 22050,
    n_fft: int = 1024,
    hop_length: int = 256,
    win_length: int = 1024,
    n_mels: int = 80,
    f_min: float = 0.0,
    f_max: float = 8000.0,
) -> np.ndarray:
    """Extract log-mel spectrogram. Input: (1, T) Tensor. Output: (n_mels, T_frames) array."""
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
    mel = transform(waveform)                         # (1, n_mels, T_frames)
    mel_db = T.AmplitudeToDB(stype="magnitude", top_db=80)(mel)
    return mel_db.squeeze(0).numpy()                  # (n_mels, T_frames)


def extract_f0(
    waveform_np: np.ndarray,
    sample_rate: int = 22050,
    hop_length: int = 256,
    fmin: float = 50.0,
    fmax: float = 1100.0,
) -> np.ndarray:
    """
    Extract F0 using librosa PYIN.
    Input: (T,) float32 numpy array.
    Output: (T_frames,) float32 array, 0.0 for unvoiced.
    """
    import librosa

    f0, _voiced_flag, _voiced_probs = librosa.pyin(
        waveform_np,
        fmin=fmin,
        fmax=fmax,
        sr=sample_rate,
        hop_length=hop_length,
    )
    return np.nan_to_num(f0, nan=0.0).astype(np.float32)


def save_features(
    mel: np.ndarray,
    f0: np.ndarray,
    mel_path: Path,
    f0_path: Path,
) -> None:
    """Save mel and F0 as .npy files."""
    mel_path.parent.mkdir(parents=True, exist_ok=True)
    f0_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(mel_path, mel)
    np.save(f0_path, f0)


def load_features(mel_path: Path, f0_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load cached mel and F0 arrays."""
    return np.load(mel_path), np.load(f0_path)
```

---

### Step 5 — `manifest.py`

```python
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class AudioEntry:
    id: str
    orig_path: str
    norm_path: str
    mel_path: str
    f0_path: str
    orig_sr: int
    target_sr: int
    orig_duration_s: float
    trimmed_duration_s: float
    n_mel_frames: int
    peak_dbfs: float


def save_manifest(entries: list[AudioEntry], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "total_files": len(entries),
        "entries": [asdict(e) for e in entries],
    }
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
```

---

### Step 6 — `cli.py`

```python
from __future__ import annotations

import argparse
from pathlib import Path

from wavekit.features import extract_f0, extract_mel, save_features
from wavekit.io import get_info, load_mono_tensor, save_wav, scan_dir
from wavekit.manifest import AudioEntry, save_manifest
from wavekit.normalize import process


def _process_file(
    path: Path,
    out_dir: Path,
    target_sr: int,
    n_mels: int,
    hop_length: int,
) -> AudioEntry | None:
    try:
        orig_sr, _channels, orig_duration = get_info(path)
        waveform, sr = load_mono_tensor(path)

        processed = process(waveform, sr, target_sr=target_sr)
        trimmed_duration = processed.shape[-1] / target_sr
        peak_dbfs = float(20 * processed.abs().max().log10() * 20)  # approx

        stem = path.stem
        norm_path = out_dir / "normalized" / f"{stem}.wav"
        mel_path = out_dir / "mel" / f"{stem}.npy"
        f0_path = out_dir / "f0" / f"{stem}.npy"

        # Save normalized WAV
        save_wav(norm_path, processed.squeeze(0).numpy(), target_sr)

        # Skip feature extraction if cached
        if not (mel_path.exists() and f0_path.exists()):
            mel = extract_mel(processed, target_sr, n_mels=n_mels, hop_length=hop_length)
            f0 = extract_f0(processed.squeeze(0).numpy(), target_sr, hop_length)
            save_features(mel, f0, mel_path, f0_path)
        else:
            mel, _ = __import__("wavekit.features", fromlist=["load_features"]).load_features(
                mel_path, f0_path
            )

        return AudioEntry(
            id=stem,
            orig_path=str(path),
            norm_path=str(norm_path),
            mel_path=str(mel_path),
            f0_path=str(f0_path),
            orig_sr=orig_sr,
            target_sr=target_sr,
            orig_duration_s=orig_duration,
            trimmed_duration_s=trimmed_duration,
            n_mel_frames=mel.shape[-1],
            peak_dbfs=peak_dbfs,
        )
    except Exception as exc:
        print(f"[warn] Failed {path.name}: {exc}")
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Audio feature extractor")
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--sample-rate", type=int, default=22050)
    parser.add_argument("--n-mels", type=int, default=80)
    parser.add_argument("--hop-length", type=int, default=256)
    args = parser.parse_args()

    paths = scan_dir(args.input_dir)
    if not paths:
        print("No audio files found.")
        return
    print(f"Found {len(paths)} files. Processing...")

    entries = []
    for path in paths:
        result = _process_file(path, args.output_dir, args.sample_rate, args.n_mels, args.hop_length)
        if result:
            entries.append(result)
        print(f"  ✓ {path.name}: {result.n_mel_frames} frames, {result.trimmed_duration_s:.2f}s" if result else f"  ✗ {path.name}")

    save_manifest(entries, args.output_dir / "manifest.json")
    print(f"\nDone. {len(entries)}/{len(paths)} files processed. Manifest at {args.output_dir}/manifest.json")
```

---

### Step 7 — Tests: `conftest.py` + `test_features.py`

```python
# tests/conftest.py
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

SR = 22050
N_FILES = 10


@pytest.fixture(scope="session")
def audio_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    d = tmp_path_factory.mktemp("audio")
    n = int(SR * 2.0)
    for i in range(N_FILES):
        t = np.linspace(0, 2.0, n, endpoint=False)
        freq = 200 + i * 50
        wave = (0.4 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
        sf.write(str(d / f"file_{i:03d}.wav"), wave, SR)
    return d
```

```python
# tests/test_features.py
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from wavekit.features import extract_mel, extract_f0
from wavekit.io import load_mono_tensor


def test_mel_shape(audio_dir: Path) -> None:
    waveform, sr = load_mono_tensor(audio_dir / "file_000.wav")
    mel = extract_mel(waveform, sr)
    assert mel.ndim == 2
    assert mel.shape[0] == 80           # n_mels
    assert mel.shape[1] > 0            # at least one frame


def test_mel_f0_alignment(audio_dir: Path) -> None:
    """Mel and F0 must have the same number of time frames."""
    waveform, sr = load_mono_tensor(audio_dir / "file_001.wav")
    mel = extract_mel(waveform, sr, hop_length=256)
    f0 = extract_f0(waveform.squeeze(0).numpy(), sr, hop_length=256)
    assert mel.shape[1] == f0.shape[0], (
        f"Mel frames {mel.shape[1]} != F0 frames {f0.shape[0]}"
    )


def test_f0_no_nans(audio_dir: Path) -> None:
    """F0 array must not contain NaN after extraction."""
    waveform, sr = load_mono_tensor(audio_dir / "file_002.wav")
    f0 = extract_f0(waveform.squeeze(0).numpy(), sr)
    assert not np.any(np.isnan(f0)), "F0 array contains NaN values"


def test_mel_deterministic(audio_dir: Path) -> None:
    waveform, sr = load_mono_tensor(audio_dir / "file_003.wav")
    mel1 = extract_mel(waveform, sr)
    mel2 = extract_mel(waveform, sr)
    np.testing.assert_array_equal(mel1, mel2)
```

---

### Step 8 — Generate test audio and run

```bash
python -c "
import numpy as np, soundfile as sf, pathlib
d = pathlib.Path('sample_audio'); d.mkdir(exist_ok=True)
sr = 22050
for i, freq in enumerate([220, 330, 440, 880]):
    t = np.linspace(0, 3.0, 3*sr, endpoint=False)
    sf.write(str(d / f'tone_{freq}hz.wav'), (0.4*np.sin(2*3.14159*freq*t)).astype('float32'), sr)
print('Created 4 test files.')
"

wavekit sample_audio/ output/ --sample-rate 22050 --n-mels 80
```

**Verify the output:**
```bash
# output/ should contain:
#   normalized/tone_220hz.wav, tone_330hz.wav, ...
#   mel/tone_220hz.npy, ...
#   f0/tone_220hz.npy, ...
#   manifest.json

python -c "import numpy as np; m = np.load('output/mel/tone_440hz.npy'); print(m.shape)"
# Expected: (80, ~258) for a 3-second clip
```

---

### Completion Checklist

- [ ] Load any `.wav` with `torchaudio.load` and explain the output shape `(C, T)` without looking at notes
- [ ] Load the same file with `sf.read` and explain why the shape is different
- [ ] Write `extract_mel` from memory — correct parameters, correct shape output
- [ ] Explain why `np.nan_to_num(f0, nan=0.0)` is required after `librosa.pyin`
- [ ] Explain the `scope="session"` fixture decision: why not `scope="function"`?
- [ ] Explain why `audio_to_wav_bytes` uses `* 32767` and not `* 32768`
- [ ] Run `pytest tests/` in `wavekit` with zero failures
- [ ] Given any `(1, T)` Tensor, write a one-liner to get a `(T,)` numpy array (three ways: `.squeeze(0).numpy()`, `.squeeze().numpy()`, `[0].numpy()`)
- [ ] Explain when you would prefer `soundfile.write` over `torchaudio.save`