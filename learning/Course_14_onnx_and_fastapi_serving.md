# Course 14: ONNX + FastAPI — Model Serving

**Prerequisites:** Course 4 (Deep Learning with PyTorch) — `nn.Module`, tensor shapes. Course 13 (Hydra + Pydantic) — Pydantic `BaseModel`, `Field`.

**What this course covers:** The serving layer converts a trained PyTorch model into a production HTTP API. By the end you will understand every file in `src/tts_pipeline/serving/` and `scripts/export_model.py` — how ONNX exports work, how `onnxruntime` runs inference, how FastAPI wires routes together, how streaming audio is delivered, and how shadow inference works.

**Project anchors:**
- [scripts/export_model.py](../scripts/export_model.py) — `export_acoustic`, `export_vocoder`, `quantize_onnx`, `validate_onnx`
- [src/tts_pipeline/serving/inference_engine.py](../src/tts_pipeline/serving/inference_engine.py) — `ONNXInferenceEngine`
- [src/tts_pipeline/serving/app.py](../src/tts_pipeline/serving/app.py) — `create_app`, lifespan
- [src/tts_pipeline/serving/schemas.py](../src/tts_pipeline/serving/schemas.py) — request/response Pydantic models
- [src/tts_pipeline/serving/routes/tts.py](../src/tts_pipeline/serving/routes/tts.py) — `/synthesize`, `/synthesize/wav`, shadow
- [src/tts_pipeline/serving/routes/tts_stream.py](../src/tts_pipeline/serving/routes/tts_stream.py) — `/synthesize_stream`
- [src/tts_pipeline/serving/routes/health.py](../src/tts_pipeline/serving/routes/health.py) — `/health`, `/ready`
- [src/tts_pipeline/serving/middleware.py](../src/tts_pipeline/serving/middleware.py) — `RequestLoggingMiddleware`

---

## Module 1 — Why ONNX for Serving

### The problem with PyTorch at inference time

Running a PyTorch model in production has costs:
- PyTorch adds ~1 GB of runtime dependencies
- Python's GIL limits true concurrency for CPU-bound inference
- The `autograd` computation graph is built and then discarded every forward pass — pure overhead at inference
- PyTorch graphs are not inherently portable across frameworks

### What ONNX is

ONNX (Open Neural Network Exchange) is a graph format — a serialized, framework-agnostic description of a neural network as a computation graph of mathematical operations (MatMul, Conv, Softmax, etc.).

```
PyTorch model (code + weights)
    ↓ torch.onnx.export()
ONNX file (.onnx) = computation graph + weights in one file
    ↓ onnxruntime.InferenceSession
Optimized inference (CPU/GPU, no PyTorch needed)
```

### What `onnxruntime` gives you

| Feature | Detail |
|---|---|
| Framework-free | No PyTorch at serving time — just `onnxruntime` |
| Graph optimization | Operator fusion, constant folding, memory planning |
| Provider backends | CPU, CUDA, TensorRT, DirectML, CoreML |
| int8 quantization | 4× smaller model, 2-4× faster CPU inference |
| Deterministic | Same inputs always produce same outputs |

In our pipeline, the export step (`dvc repro export`) runs once after training. The serving container imports only `onnxruntime`, not `torch`, `lightning`, or any training dependency.

---

## Module 2 — Exporting PyTorch to ONNX

### `torch.onnx.export()` — the acoustic model

```python
def export_acoustic(model: torch.nn.Module, export_path: Path, n_mels: int = 80) -> None:
    model.eval()                                  # export_model.py:11
    B, T_text = 1, 20
    dummy_ids     = torch.zeros(B, T_text, dtype=torch.long)
    dummy_lengths = torch.tensor([T_text], dtype=torch.long)

    torch.onnx.export(
        model,
        (dummy_ids, dummy_lengths),               # tuple of all inputs
        str(export_path),                         # output file path
        opset_version=17,                         # export_model.py:21
        input_names=["phoneme_ids", "phoneme_lengths"],
        output_names=["mel"],
        dynamic_axes={                            # export_model.py:23
            "phoneme_ids":     {0: "batch", 1: "text_len"},
            "phoneme_lengths": {0: "batch"},
            "mel":             {0: "batch", 2: "mel_len"},
        },
        do_constant_folding=True,
    )
```

### What each argument does

**`model.eval()`** — essential. Disables dropout and BatchNorm running statistics updates. An ONNX export with `model.train()` would bake random dropout masks into the graph.

**Dummy inputs** — ONNX export works by tracing: PyTorch runs one forward pass with the dummy inputs, records every operation, and saves the trace. The dummy shapes define the static dimensions; `dynamic_axes` marks which dimensions can vary at runtime.

**`opset_version=17`** — ONNX opset is the version of the operator specification. Higher versions support more operators. Opset 17 (released 2022) covers everything our models use. This must match what `onnxruntime` supports — check with `onnxruntime.get_device()`.

**`input_names` / `output_names`** — string names for the graph's input and output nodes. These become the keys you pass to `session.run()` at inference time. Without naming them, ONNX generates opaque names like `"onnx::Gemm_0"`.

**`dynamic_axes`** — the critical argument for variable-length sequences. Without it, ONNX bakes the dummy shape into the graph and rejects inputs of any other size.

```python
# Without dynamic_axes:
# graph accepts ONLY shape (1, 20) for phoneme_ids — all other lengths fail
# With dynamic_axes={..., 1: "text_len"}:
# dimension 1 is symbolic "text_len" — any length accepted
```

Axis 0 (`"batch"`) is made dynamic on all inputs so the model can process batch_size ≠ 1. Axis 1 (`"text_len"`) on `phoneme_ids` allows different phoneme sequence lengths. Axis 2 (`"mel_len"`) on the output mel allows the predicted mel length to vary.

**`do_constant_folding=True`** — pre-computes subgraphs whose inputs are constants. For example, sinusoidal position encodings computed from static position indices are evaluated once during export and baked as constants. This reduces runtime computation.

### Exporting the vocoder

```python
def export_vocoder(model, export_path, n_mels=80) -> None:   # export_model.py:33
    model.eval()
    dummy_mel = torch.zeros(1, n_mels, 100)    # (batch, n_mels, mel_len)
    torch.onnx.export(
        model,
        dummy_mel,                             # single tensor input (not a tuple)
        str(export_path),
        opset_version=17,
        input_names=["mel"],
        output_names=["audio"],
        dynamic_axes={
            "mel":   {0: "batch", 2: "mel_len"},
            "audio": {0: "batch", 2: "audio_len"},
        },
        do_constant_folding=True,
    )
```

The vocoder's single input is a mel spectrogram `(1, n_mels=80, T_mel)`. The output `audio` has shape `(1, 1, T_audio)` where `T_audio = T_mel × hop_length = T_mel × 256`. Both mel and audio time dimensions are dynamic.

Note: `vocoder.remove_weight_norm()` is called before export (export_model.py:107). Weight normalization is a training trick — it splits one weight matrix into magnitude + direction. Removing it folds the normalization into the weight permanently, giving a simpler and faster graph.

---

## Module 3 — Quantization and Validation

### `quantize_dynamic` — int8 quantization

```python
def quantize_onnx(onnx_path: Path) -> Path:            # export_model.py:52
    from onnxruntime.quantization import QuantType, quantize_dynamic

    q_path = onnx_path.with_suffix(".int8.onnx")
    quantize_dynamic(str(onnx_path), str(q_path), weight_type=QuantType.QInt8)
    return q_path
```

**Dynamic quantization** converts the model's weights from float32 to int8. Activations are quantized to int8 at runtime per-tensor (the scale is computed dynamically from each batch). This is called "dynamic" because activation scales are not fixed at quantization time.

The difference from static quantization:
- **Dynamic** — weights pre-quantized, activations quantized at runtime. Fast to apply, works without a calibration dataset.
- **Static** — both weights and activations pre-quantized using a calibration dataset. Faster inference but requires representative data.

`QuantType.QInt8` uses signed 8-bit integers (range -128 to 127). The alternative is `QUInt8` (0 to 255). QInt8 is better for models with symmetric weight distributions (most transformers).

### Practical effect

```
acoustic.onnx (fp32):   ~180 MB
acoustic.int8.onnx:     ~45 MB   (4× smaller)
CPU inference speedup:  1.5-3×   (depends on CPU support for int8 matmul)
Quality loss:           <1% on MOS score for most TTS models
```

### `validate_onnx` — correctness check

```python
def validate_onnx(onnx_path: Path) -> None:            # export_model.py:61
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    print(f"ONNX validation OK: inputs={[i.name for i in sess.get_inputs()]}")
```

This does not check output correctness — it verifies that ONNX Runtime can load the file and create a session without errors. It catches:
- Malformed ONNX graphs
- Unsupported operators for the given opset
- Missing weights

For correctness validation, you'd compare outputs from the PyTorch model and the ONNX session on the same input.

---

## Module 4 — Running Inference with ONNX Runtime

### `InferenceSession` setup

```python
opts = ort.SessionOptions()
opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL   # inference_engine.py:42

providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

self._acoustic = ort.InferenceSession(str(acoustic_path), opts, providers=providers)
```

**`graph_optimization_level`** — the optimization pipeline applied when loading the session:

| Level | What it does |
|---|---|
| `ORT_DISABLE_ALL` | No optimizations (debug only) |
| `ORT_ENABLE_BASIC` | Constant folding, redundant node elimination |
| `ORT_ENABLE_EXTENDED` | Operator fusion (e.g., fuse MatMul + Add + Relu into one op) |
| `ORT_ENABLE_ALL` | All above + layout transformation, memory planning |

`ORT_ENABLE_ALL` maximizes performance at the cost of a slightly longer load time (typically seconds). Always use it for production.

**Provider priority list** — `onnxruntime` tries providers left to right. If CUDA is available, it runs on GPU. If not, it falls back to CPU. No code change is needed to switch between GPU and CPU — just install `onnxruntime-gpu` and have CUDA available.

### Running inference

```python
mel_out = self._acoustic.run(
    None,                                              # None = return all outputs
    {
        "phoneme_ids":     phoneme_array,              # np.ndarray, dtype=int64
        "phoneme_lengths": phoneme_lengths,            # np.ndarray, dtype=int64
    },
)[0]   # returns list of arrays; [0] = first output ("mel")
```

**Key points:**
- Inputs are **NumPy arrays**, not PyTorch tensors. `onnxruntime` has no PyTorch dependency.
- Input dtype must exactly match what was exported. `phoneme_ids` needs `np.int64` because the embedding layer expects `torch.long` (= int64).
- `session.run(None, ...)` returns a Python list of NumPy arrays, one per output node. `[0]` gets `mel`.
- Output shape: `(1, n_mels=80, T_mel)` — a numpy array, not a tensor.

### Full inference trace in `synthesize()`

```python
def synthesize(self, text, n_steps=50, length_scale=1.0):   # inference_engine.py:58
    clean = clean_text(text)
    phoneme_ids = self._tokenizer.encode(clean.split())
    phoneme_array  = np.array([phoneme_ids], dtype=np.int64)      # (1, T_text)
    phoneme_lengths = np.array([len(phoneme_ids)], dtype=np.int64) # (1,)

    t0 = time.perf_counter()

    mel_out = self._acoustic.run(                             # acoustic ONNX
        None,
        {"phoneme_ids": phoneme_array, "phoneme_lengths": phoneme_lengths},
    )[0]  # (1, 80, T_mel)

    audio_out = self._vocoder.run(None, {"mel": mel_out})[0]  # (1, 1, T_audio)

    elapsed = time.perf_counter() - t0
    audio = audio_out.squeeze()                                # (T_audio,)
    duration_sec = audio.shape[0] / self._sample_rate
    rtf = elapsed / max(duration_sec, 1e-8)                   # Real-Time Factor

    wav_bytes = audio_to_wav_bytes(audio, self._sample_rate)
    return wav_bytes, rtf
```

### RTF — Real-Time Factor

```python
rtf = elapsed / max(duration_sec, 1e-8)   # inference_engine.py:84
```

RTF = (inference time) / (audio duration). Interpretation:
- RTF = 0.05 → inference takes 5% of real-time → 20× faster than real-time
- RTF = 0.20 → inference takes 20% of real-time → the `rtf_p95` absolute floor from the eval gate
- RTF = 1.0  → inference takes as long as playing the audio → not usable for real-time
- RTF > 1.0  → slower than real-time → unacceptable for production

The `max(..., 1e-8)` guards against division by zero for very short synthesized utterances.

---

## Module 5 — FastAPI Application Structure

### Why FastAPI

FastAPI is an async Python web framework built on Starlette and Pydantic. Key properties for our use case:

| Feature | Why it matters |
|---|---|
| Async (ASGI) | Multiple requests handled without blocking |
| Pydantic request/response models | Automatic validation + OpenAPI docs generation |
| Type hints → docs | `/docs` endpoint is auto-generated |
| `StreamingResponse` | Native support for chunked audio streaming |
| `app.state` | Shared mutable state (the inference engine) |

### `create_app()` — the application factory

```python
def create_app(bundle_path, shadow_bundle_path=None) -> FastAPI:  # app.py:17

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        app.state.engine = ONNXInferenceEngine(Path(bundle_path))
        if shadow_bundle_path:
            app.state.shadow_engine = ONNXInferenceEngine(Path(shadow_bundle_path))
        else:
            app.state.shadow_engine = None
        yield                                    # ← application runs here
        logger.info("Shutting down inference engine")

    app = FastAPI(title="TTS Pipeline API", lifespan=lifespan)
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(CORSMiddleware, allow_origins=["*"], ...)
    app.include_router(health.router, tags=["health"])
    app.include_router(tts.router, tags=["synthesis"])
    app.include_router(tts_stream.router, tags=["synthesis"])
    return app
```

### The `lifespan` context manager

The `lifespan` pattern (FastAPI 0.93+) replaces the old `@app.on_event("startup")` / `@app.on_event("shutdown")` hooks. Everything before `yield` runs at startup; the `yield` suspends until shutdown; everything after `yield` runs at shutdown.

Why load the engine in `lifespan` rather than at module level?
- Module-level code runs at import time — the ONNX files may not exist yet
- If loading fails, you want a clean error at startup, not a cryptic `ImportError`
- The `lifespan` pattern is the FastAPI-idiomatic way to manage shared application state

### `app.state` — sharing the engine across requests

```python
# Set in lifespan:
app.state.engine = ONNXInferenceEngine(bundle_path)

# Read in route handlers:
engine = getattr(request.app.state, "engine", None)   # tts.py:15
```

`app.state` is a `State` object that acts as a simple attribute bag. It is shared across all requests. `getattr(..., None)` returns `None` instead of raising `AttributeError` if the engine failed to load.

### Application factory vs module-level `app`

The project uses `create_app()` (a factory) rather than a module-level `app = FastAPI()`. This makes testing easy:

```python
# In tests:
from tts_pipeline.serving.app import create_app
from fastapi.testclient import TestClient

app = create_app(bundle_path="test_bundle/")
client = TestClient(app)
response = client.get("/health")
```

The factory pattern lets you create multiple app instances with different configs — for example, one pointing to the production bundle and one to the shadow bundle in tests.

---

## Module 6 — Pydantic Request/Response Schemas

### `SynthesizeRequest`

```python
class SynthesizeRequest(BaseModel):                   # schemas.py:6
    text: str = Field(..., min_length=1, max_length=500)
    speaker_id: int | None = Field(default=None)
    length_scale: float = Field(default=1.0, ge=0.5, le=2.0)
    n_steps: int = Field(default=50, ge=1, le=200)

    @field_validator("text")
    @classmethod
    def text_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must not be blank")
        return v.strip()
```

**`Field(...)` arguments:**
- `...` (Ellipsis) — required field, no default
- `min_length=1, max_length=500` — validated by Pydantic before the route handler runs
- `ge=0.5, le=2.0` — `ge` = greater-or-equal, `le` = less-or-equal (closed bounds)
- `ge=1, le=200` — `n_steps` must be between 1 and 200 inclusive

**`field_validator("text")`** runs after `min_length=1` passes. A string of only spaces (`"   "`) passes `min_length=1` but fails `v.strip()`. The validator strips the text, so `body.text` is always stripped when it reaches the route handler.

### Why validate at the schema level

If validation fails (e.g., `text` is 600 characters, or `length_scale=5.0`), FastAPI returns HTTP 422 with a detailed error body **before** the route handler runs. The inference engine is never called.

```json
{
  "detail": [
    {
      "type": "string_too_long",
      "loc": ["body", "text"],
      "msg": "String should have at most 500 characters",
      "input": "very long text..."
    }
  ]
}
```

### `SynthesizeResponse`

```python
class SynthesizeResponse(BaseModel):      # schemas.py:20
    audio_bytes_b64: str                  # Base64-encoded WAV bytes
    duration_sec: float
    rtf: float
    sample_rate: int = 22050
```

The audio is returned as Base64-encoded WAV bytes because JSON does not support raw binary. The client decodes with `base64.b64decode(response["audio_bytes_b64"])` to get the WAV bytes.

For direct browser playback, the `/synthesize/wav` endpoint returns raw bytes with `Content-Type: audio/wav` instead.

---

## Module 7 — Route Handlers

### `/synthesize` — JSON response with Base64 audio

```python
@router.post("/synthesize", response_model=SynthesizeResponse)   # tts.py:13
async def synthesize(request: Request, body: SynthesizeRequest) -> SynthesizeResponse:
    engine = getattr(request.app.state, "engine", None)
    if engine is None or not engine.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        wav_bytes, rtf = engine.synthesize(
            body.text, n_steps=body.n_steps, length_scale=body.length_scale
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    shadow_engine = getattr(request.app.state, "shadow_engine", None)
    if shadow_engine is not None and shadow_engine.is_loaded:
        asyncio.create_task(_run_shadow(shadow_engine, body.text, body.n_steps))

    duration_sec = len(wav_bytes) / (engine.sample_rate * 2)   # 16-bit PCM
    return SynthesizeResponse(
        audio_bytes_b64=base64.b64encode(wav_bytes).decode(),
        duration_sec=duration_sec,
        rtf=rtf,
        sample_rate=engine.sample_rate,
    )
```

**503 vs 500:**
- 503 Service Unavailable — model not loaded (startup failure, bundle missing). The client should not retry immediately.
- 500 Internal Server Error — model loaded but inference failed. May be retryable.

**`duration_sec` computation:**

```python
duration_sec = len(wav_bytes) / (engine.sample_rate * 2)
```

WAV bytes contain 16-bit PCM (`int16` = 2 bytes per sample). At 22050 Hz: each second of audio = 22050 × 2 = 44100 bytes.

### `/synthesize/wav` — raw WAV bytes

```python
@router.post("/synthesize/wav")                    # tts.py:45
async def synthesize_wav(request: Request, body: SynthesizeRequest) -> Response:
    wav_bytes, _ = engine.synthesize(body.text, ...)
    return Response(content=wav_bytes, media_type="audio/wav")
```

`Response(content=bytes, media_type="audio/wav")` sends raw bytes with the correct MIME type. Browsers and media players can receive this directly. The `_` discards the RTF — not logged on this endpoint.

### `/health` and `/ready`

```python
@router.get("/health", response_model=HealthResponse)    # health.py:10
async def health() -> HealthResponse:
    return HealthResponse(status="ok")                   # always 200

@router.get("/ready", response_model=ReadyResponse)      # health.py:15
async def ready(request: Request) -> ReadyResponse:
    engine = getattr(request.app.state, "engine", None)
    loaded = engine is not None and engine.is_loaded
    return ReadyResponse(ready=loaded, model_loaded=loaded, shadow_loaded=...)
```

Two separate endpoints serve two different purposes:
- **`/health`** — always 200 if the process is running. Used by container orchestration (Kubernetes) to know the process is alive.
- **`/ready`** — 200 only if the model is loaded. Used to know whether the pod should receive traffic. Kubernetes calls `/ready` before routing requests to a new pod.

---

## Module 8 — Streaming Audio

### Why streaming

For a long text (10 sentences), the user waits for the entire synthesis before hearing any audio. Sentence-by-sentence streaming delivers the first audio chunk ~200ms after the request, with subsequent sentences following in a pipeline.

### `StreamingResponse` and the generator pattern

```python
@router.get("/synthesize_stream")                             # tts_stream.py:18
async def synthesize_stream(request: Request, text: str, ...) -> StreamingResponse:
    sentences = _split_sentences(text) if chunk_sentences else [text]

    async def audio_generator():
        yield wav_header_bytes(engine.sample_rate)            # tts_stream.py:37

        for sentence in sentences:
            mel = engine.synthesize_mel(sentence, n_steps=n_steps)
            audio_chunk = engine.vocode(mel)
            pcm = (audio_chunk.flatten() * 32767).clip(-32768, 32767).astype(np.int16)
            yield pcm.tobytes()

    return StreamingResponse(
        audio_generator(),
        media_type="audio/wav",
        headers={"X-Accel-Buffering": "no"},
    )
```

`StreamingResponse` takes an async generator. FastAPI sends each `yield`ed chunk to the client as soon as it is produced — the client can start playing audio while the server is still synthesizing later sentences.

### The WAV header trick

```python
yield wav_header_bytes(engine.sample_rate)   # tts_stream.py:37
```

A valid WAV file requires a 44-byte header that includes the total data size. Since we're streaming and don't know the total size upfront, `wav_header_bytes` writes `0` for the data size field. Most audio players handle this gracefully (they play until the stream closes).

After the header, only raw PCM chunks are yielded — no per-chunk headers.

### PCM conversion

```python
pcm = (audio_chunk.flatten() * 32767).clip(-32768, 32767).astype(np.int16)
yield pcm.tobytes()
```

The neural network output is float32 in range roughly `[-1.0, 1.0]`. 16-bit PCM (WAV format) uses integers in `[-32768, 32767]`. The conversion:
1. `* 32767` — scale to int16 range
2. `.clip(-32768, 32767)` — clamp any out-of-range values (clipping distortion > out-of-range wrapping)
3. `.astype(np.int16)` — cast to int16
4. `.tobytes()` — get the raw bytes

### `X-Accel-Buffering: no`

```python
headers={"X-Accel-Buffering": "no"}
```

This header tells Nginx (if used as a reverse proxy) not to buffer the response. Without it, Nginx accumulates the entire stream before forwarding — defeating the purpose of streaming.

### Sentence splitting

```python
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")   # tts_stream.py:10

def _split_sentences(text: str) -> list[str]:
    parts = _SENTENCE_SPLIT.split(text.strip())
    return [p.strip() for p in parts if p.strip()]
```

`(?<=[.!?])\s+` is a lookbehind assertion: split on whitespace that is preceded by `.`, `!`, or `?`. This keeps the punctuation with the sentence and splits at the space after it.

```
"Hello world. How are you? Fine." → ["Hello world.", "How are you?", "Fine."]
```

---

## Module 9 — Shadow Inference

### What shadow inference is

Shadow inference runs the candidate model (in `Staging` or `Shadow` stage) on live production traffic in parallel with the production model, but does not return the shadow output to the user. This lets you:
- Measure the shadow model's RTF on real inputs (not synthetic benchmarks)
- Compare audio quality on the same texts that real users are sending
- Detect regressions before promoting to Production

### Fire-and-forget with `asyncio.create_task`

```python
if shadow_engine is not None and shadow_engine.is_loaded:
    asyncio.create_task(                                    # tts.py:33
        _run_shadow(shadow_engine, body.text, body.n_steps)
    )
```

`asyncio.create_task()` schedules the coroutine to run in the background event loop without awaiting it. The `/synthesize` route returns the production response immediately — the shadow inference happens concurrently.

```python
async def _run_shadow(shadow_engine, text, n_steps) -> None:   # tts.py:55
    try:
        wav_bytes, rtf = shadow_engine.synthesize(text, n_steps=n_steps)
        if mlflow.active_run():
            mlflow.log_metric("shadow_rtf", rtf)
    except Exception:
        pass   # ← never let shadow failures affect production
```

`except Exception: pass` is intentional — a shadow failure must never propagate to the production response. The `pass` means "log nothing, swallow everything, continue." In a production system you'd log to a separate telemetry channel here.

---

## Module 10 — Middleware

### `RequestLoggingMiddleware`

```python
class RequestLoggingMiddleware(BaseHTTPMiddleware):      # middleware.py:13
    async def dispatch(self, request: Request, call_next) -> Response:
        t0 = time.perf_counter()
        response = await call_next(request)              # ← call the route handler
        elapsed = (time.perf_counter() - t0) * 1000
        logger.info("%s %s → %d (%.1fms)",
                    request.method, request.url.path,
                    response.status_code, elapsed)
        return response
```

`BaseHTTPMiddleware` wraps every request. The `dispatch` method calls `call_next(request)` to execute the route handler and capture the response. This pattern lets you measure total request latency including serialization, routing, and handler time.

Output example:
```
POST /synthesize → 200 (243.7ms)
GET  /health     → 200 (0.3ms)
POST /synthesize → 503 (0.1ms)   ← model not loaded
```

### `CORSMiddleware`

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],     # app.py:44
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
```

CORS (Cross-Origin Resource Sharing) headers allow browsers to make API calls from a different origin (e.g., a frontend at `localhost:3000` calling the API at `localhost:8080`). Without CORS headers, the browser blocks the request before it reaches the server.

`allow_origins=["*"]` is permissive — any origin is allowed. For production, restrict to your frontend's domain: `allow_origins=["https://your-app.com"]`.

### Middleware execution order

```python
app.add_middleware(RequestLoggingMiddleware)   # outer (wraps everything)
app.add_middleware(CORSMiddleware, ...)        # inner (closer to route)
```

Middlewares execute in reverse registration order: CORS runs first, then RequestLogging. The request path is: CORS → RequestLogging → route handler. The response path is: route handler → RequestLogging → CORS.

---

## Module 11 — The Model Bundle

### What the export stage produces

```
artifacts/export/
├── acoustic.onnx           ← int8 quantized acoustic model
├── vocoder.onnx            ← int8 quantized vocoder
├── config.json             ← sample_rate, hop_length, n_mels, n_ode_steps, language
├── phoneme_vocab.json      ← Tokenizer vocabulary (built during training)
└── model_card.json         ← version, dvc_commit, eval_metrics
```

`config.json` is the minimal config the inference engine needs:

```python
config_out = {                                   # export_model.py:120
    "sample_rate": cfg.data.sample_rate,         # 22050
    "hop_length": cfg.data.hop_length,           # 256
    "n_mels": cfg.data.n_mels,                   # 80
    "n_ode_steps": cfg.model.n_ode_steps,        # 10
    "language": cfg.data.language,               # "vi"
}
```

`model_card.json` links the ONNX files back to the commit and evaluation metrics:

```python
{
    "version": "0.1.0",
    "dvc_commit": "fd2d31f...",     # the git commit that produced these weights
    "language": "vi",
    "eval_metrics": {
        "utmos_mean": 3.42,
        "cer": 0.12,
        "stoi_mean": 0.74,
        "rtf_p95": 0.09
    }
}
```

The bundle is self-contained: copy it anywhere, point `ONNXInferenceEngine` at it, serve.

---

## Module 12 — Mini Practice Project: `servkit`

An independent mini-project that builds a complete ONNX + FastAPI serving stack for a simple regression model — no TTS pipeline code.

**Goal:** Export a sklearn `LinearRegression` to ONNX, serve it behind a FastAPI endpoint with request validation, response schema, and a `/health` + `/ready` check.

### Project structure

```
servkit/
├── export.py      # train a model and export to ONNX
├── engine.py      # OnnxRegressionEngine
├── schemas.py     # Pydantic request/response models
├── app.py         # FastAPI application factory
└── routes.py      # /predict, /health, /ready
```

### `export.py`

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from pathlib import Path


def train_and_export(output_dir: str = "bundle") -> None:
    X, y = make_regression(n_samples=500, n_features=10, noise=0.1, random_state=42)
    X = X.astype(np.float32)

    model = LinearRegression()
    model.fit(X, y)

    # Export to ONNX
    initial_type = [("input", FloatTensorType([None, 10]))]  # None = dynamic batch
    onnx_model = convert_sklearn(model, initial_types=initial_type)

    Path(output_dir).mkdir(exist_ok=True)
    with open(f"{output_dir}/regression.onnx", "wb") as f:
        f.write(onnx_model.SerializeToString())

    # Write config
    import json
    config = {"n_features": 10, "output_name": "variable"}
    with open(f"{output_dir}/config.json", "w") as f:
        json.dump(config, f)

    print(f"Exported to {output_dir}/")
    print(f"  Train MSE: {np.mean((model.predict(X) - y)**2):.4f}")


if __name__ == "__main__":
    train_and_export()
```

### `engine.py`

```python
import json
import numpy as np
from pathlib import Path


class OnnxRegressionEngine:
    def __init__(self, bundle_path: str = "bundle") -> None:
        import onnxruntime as ort

        p = Path(bundle_path)
        with open(p / "config.json") as f:
            self._config = json.load(f)

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(
            str(p / "regression.onnx"),
            opts,
            providers=["CPUExecutionProvider"],
        )
        self._loaded = True
        self._input_name = self._session.get_inputs()[0].name
        self._output_name = self._config["output_name"]

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def predict(self, features: list[float]) -> float:
        arr = np.array([features], dtype=np.float32)  # (1, n_features)
        result = self._session.run(None, {self._input_name: arr})
        return float(result[0].squeeze())
```

### `schemas.py`

```python
from pydantic import BaseModel, Field
from typing import Annotated


class PredictRequest(BaseModel):
    features: list[float] = Field(
        ...,
        min_length=10,
        max_length=10,
        description="Exactly 10 numeric features",
    )


class PredictResponse(BaseModel):
    prediction: float
    n_features: int


class HealthResponse(BaseModel):
    status: str = "ok"


class ReadyResponse(BaseModel):
    ready: bool
    model_loaded: bool
```

### `app.py`

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI
from engine import OnnxRegressionEngine
import routes


def create_app(bundle_path: str = "bundle") -> FastAPI:

    @asynccontextmanager
    async def lifespan(app):
        app.state.engine = OnnxRegressionEngine(bundle_path)
        yield

    app = FastAPI(title="Regression API", lifespan=lifespan)
    app.include_router(routes.router)
    return app


if __name__ == "__main__":
    import uvicorn
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### `routes.py`

```python
from fastapi import APIRouter, HTTPException, Request
from schemas import HealthResponse, PredictRequest, PredictResponse, ReadyResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse()


@router.get("/ready", response_model=ReadyResponse)
async def ready(request: Request):
    engine = getattr(request.app.state, "engine", None)
    loaded = engine is not None and engine.is_loaded
    return ReadyResponse(ready=loaded, model_loaded=loaded)


@router.post("/predict", response_model=PredictResponse)
async def predict(request: Request, body: PredictRequest):
    engine = getattr(request.app.state, "engine", None)
    if engine is None or not engine.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        pred = engine.predict(body.features)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return PredictResponse(prediction=pred, n_features=len(body.features))
```

### Running it

```bash
pip install fastapi uvicorn onnxruntime scikit-learn skl2onnx numpy

# Export
python export.py

# Serve
python app.py

# Test (in another terminal)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]}'

# Health checks
curl http://localhost:8000/health
curl http://localhost:8000/ready

# Auto-generated docs
open http://localhost:8000/docs
```

### Exercises

1. **Inspect the ONNX graph** — after `python export.py`, run:
   ```python
   import onnx
   model = onnx.load("bundle/regression.onnx")
   print(onnx.helper.printable_graph(model.graph))
   ```
   Identify the `MatMul` and `Add` nodes that implement `y = Xw + b`.

2. **Add request timing middleware** — copy the `RequestLoggingMiddleware` pattern into `app.py`. Verify that every request prints method, path, status code, and elapsed ms.

3. **Batch prediction** — change `PredictRequest` to accept a `list[list[float]]` (a batch of feature vectors) and return `list[float]`. Update `engine.predict()` to accept `(N, 10)` arrays and return a list of predictions.

4. **Trigger a 422 error** — send `{"features": [1.0, 2.0]}` (only 2 features). Observe the Pydantic validation error. Find the JSON field that says `"msg": "List should have at least 10 items"`.

5. **Add quantization** — install `onnxruntime` quantization utilities and apply `quantize_dynamic` to `bundle/regression.onnx`. Compare file sizes before and after. Does the prediction change? (For a simple linear model the quantization effect is negligible.)

6. **Add a `/metrics` endpoint** — maintain a counter of total predictions and a running mean of prediction values in `app.state`. Add a `GET /metrics` route that returns these values. Verify the counter increments with each request.

---

## Key Takeaways

- ONNX export works by tracing one forward pass with dummy inputs. `dynamic_axes` marks which tensor dimensions can vary at runtime — always set it for batch, sequence length, and output length dimensions.
- `do_constant_folding=True` pre-computes static subgraphs (like positional encodings) at export time. `remove_weight_norm()` before export folds normalization into weights.
- `quantize_dynamic` converts weights to int8, giving ~4× smaller files and 1.5-3× faster CPU inference with minimal quality loss.
- `ort.InferenceSession` takes a provider list in priority order — `["CUDAExecutionProvider", "CPUExecutionProvider"]` auto-selects GPU when available.
- `session.run(None, {"input_name": np_array})` returns a Python list of NumPy arrays. Inputs must be NumPy (not tensors) with the correct dtype.
- FastAPI's `lifespan` context manager is the correct place to load the model — everything before `yield` is startup, after `yield` is shutdown.
- `app.state` shares the inference engine across all requests. Always guard with `getattr(..., None)` and check `is_loaded`.
- `StreamingResponse` with an async generator streams chunks to the client as they are produced. Yield the WAV header first (data size = 0), then raw PCM chunks.
- Shadow inference uses `asyncio.create_task()` for fire-and-forget parallel execution. Always wrap in `try/except: pass` — shadow failures must never affect production responses.
- `/health` (always 200 if process alive) and `/ready` (200 only if model loaded) serve different purposes for container orchestration health checks.