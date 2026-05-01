# Course 5: PyTorch Lightning for Training Orchestration

**Prerequisites:** Course 4 (Deep Learning with PyTorch) — you should be comfortable with `nn.Module`, the 3-step training loop, `DataLoader`, and schedulers.

**What this course covers:** PyTorch Lightning is the training framework that wraps the raw PyTorch loop in our pipeline. By the end you will understand every flag in `run_training()`, every hook called during `trainer.fit()`, and how `LightningModule`, `LightningDataModule`, and `Callback` each take a responsibility that otherwise lives in ad-hoc training scripts.

**Project anchors:**
- [src/tts_pipeline/models/matcha/model.py](../src/tts_pipeline/models/matcha/model.py) — `MatchaTTSLightning`
- [src/tts_pipeline/training/trainer.py](../src/tts_pipeline/training/trainer.py) — `run_training()`
- [src/tts_pipeline/training/callbacks.py](../src/tts_pipeline/training/callbacks.py) — `AudioLoggerCallback`, `MLflowMetricsCallback`
- [src/tts_pipeline/data/dataset.py](../src/tts_pipeline/data/dataset.py) — `TTSDataset`, `TTSDataModule`, `collate_fn`

---

## Module 1 — Why Lightning Exists

### The raw-PyTorch training loop problem

A production training script without a framework accumulates boilerplate that has nothing to do with your model:

```python
# Pure PyTorch — every project reinvents this
for epoch in range(max_epochs):
    model.train()
    for batch in train_loader:
        batch = {k: v.to(device) for k, v in batch.items()}   # device transfer
        optimizer.zero_grad()
        loss = model(**batch)["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
    
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            val_loss += model(**batch)["loss"].item()
    
    # checkpoint saving, early stopping, logging — all manual
```

Every line that isn't `loss = model(...)` is infrastructure. Lightning separates:

| Concern | Lightning class |
|---|---|
| Model + losses | `LightningModule` |
| Data loading | `LightningDataModule` |
| Infrastructure (device, precision, logging) | `Trainer` |
| Side effects (audio logging, metric sync) | `Callback` |

The Trainer calls your `LightningModule` hooks in the right order; you never write `model.to(device)` or `optimizer.zero_grad()` yourself.

### Where Lightning sits in the project

```
run_training()                    ← trainer.py:15
  ├─ MatchaTTSLightning           ← model.py:118
  │   ├─ training_step()          ← model.py:130
  │   ├─ validation_step()        ← model.py:140
  │   └─ configure_optimizers()   ← model.py:149
  ├─ TTSDataModule                ← dataset.py:90
  │   ├─ setup()                  ← dataset.py:105
  │   ├─ train_dataloader()       ← dataset.py:121
  │   └─ val_dataloader()         ← dataset.py:132
  ├─ ModelCheckpoint              ← trainer.py:54
  ├─ EarlyStopping                ← trainer.py:61
  ├─ AudioLoggerCallback          ← callbacks.py:20
  ├─ MLflowMetricsCallback        ← callbacks.py:46
  └─ L.Trainer(...)               ← trainer.py:67
       └─ trainer.fit(pl_model, datamodule=data_module)
```

---

## Module 2 — LightningModule

### The three mandatory hooks

Every `LightningModule` must implement (at minimum) these three methods. The Trainer calls them in a fixed order.

```python
class MatchaTTSLightning(L.LightningModule):   # model.py:118
    def training_step(self, batch, batch_idx):
        # Called for every training batch.
        # Must return the loss tensor (scalar).
        ...
        return loss

    def validation_step(self, batch, batch_idx):
        # Called for every validation batch.
        # Return value is ignored by Trainer (use self.log instead).
        ...

    def configure_optimizers(self):
        # Called once before training starts.
        # Returns optimizer + (optionally) scheduler config.
        ...
```

### `training_step` in detail

```python
def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
    out = self.model(
        batch["phoneme_ids"],   # already on correct device — Trainer moved it
        batch["phoneme_len"],
        batch["mel"],
        batch["mel_len"],
    )
    self.log_dict({f"train/{k}": v for k, v in out.items()}, prog_bar=True)
    return out["loss"]
```

Key observations:
- **`batch` is already on the device.** The Trainer calls `batch = batch.to(self.device)` before your hook. You never write `.cuda()` or `.to(device)`.
- **Gradient zero, backward, and optimizer step are handled by Trainer.** You just return the loss.
- **`self.log_dict()`** queues metrics for the configured logger (MLflow in our case). The `prog_bar=True` flag shows them in the tqdm bar.

### `validation_step` in detail

```python
def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
    out = self.model(...)
    self.log_dict({f"val/{k}": v for k, v in out.items()}, prog_bar=True)
```

The Trainer automatically:
1. Calls `model.eval()` before the validation loop
2. Wraps the loop in `torch.no_grad()`
3. Calls `model.train()` after

You never write these lines yourself.

### `self.log()` and `self.log_dict()`

```python
# Log a single scalar
self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)

# Log multiple scalars at once (equivalent to calling log() for each)
self.log_dict({"train/loss": loss, "train/flow_loss": flow_loss})
```

| Parameter | Default | Meaning |
|---|---|---|
| `on_step` | True in training | Log per batch step |
| `on_epoch` | True in validation | Log mean across the epoch |
| `prog_bar` | False | Show in tqdm bar |
| `sync_dist` | False | Average across GPUs (set True for multi-GPU val) |

### `save_hyperparameters()`

```python
def __init__(self, model_cfg: dict, training_cfg: dict) -> None:
    super().__init__()
    self.save_hyperparameters()           # model.py:123
    self.model = MatchaTTS(**model_cfg)
```

`save_hyperparameters()` stores all `__init__` arguments in `self.hparams` and serializes them into the checkpoint. This is what makes `load_from_checkpoint()` work without you manually passing arguments again.

```python
# Deserialize: hparams are read from checkpoint, __init__ is called automatically
model = MatchaTTSLightning.load_from_checkpoint("best_model.ckpt")
```

---

## Module 3 — configure_optimizers

### The return contract

`configure_optimizers` can return several shapes. Our pipeline returns a dictionary:

```python
def configure_optimizers(self) -> Any:            # model.py:149
    opt = AdamW(self.parameters(), lr=..., weight_decay=1e-2)
    warmup = LinearLR(opt, start_factor=0.1, total_iters=1000)
    cosine = CosineAnnealingLR(opt, T_max=1000)
    scheduler = SequentialLR(opt, schedulers=[warmup, cosine], milestones=[1000])
    return {
        "optimizer": opt,
        "lr_scheduler": {
            "scheduler": scheduler,
            "interval": "step",     # ← step after every batch, not every epoch
        },
    }
```

The `"interval": "step"` key is critical. Without it Lightning defaults to `"epoch"`, which means warmup would tick once per epoch — taking 1000 epochs instead of 1000 steps.

### Scheduler phasing

```
Step 0-999: LinearLR (warmup)
  lr = base_lr × (0.1 + (1-0.1) × step/1000)
  At step 0:   lr = 1e-5
  At step 999: lr ≈ 1e-4

Step 1000+: CosineAnnealingLR
  lr = 1e-4 × (1 + cos(π × epoch/T_max)) / 2
  Decays smoothly to ~0 over T_max=1000 epochs
```

This is identical to what Course 4 covered — the only difference is that Lightning reads this config and calls `scheduler.step()` for you at the correct interval.

---

## Module 4 — LightningDataModule

### Lifecycle methods

A `LightningDataModule` has four lifecycle hooks:

```python
class TTSDataModule(L.LightningDataModule):   # dataset.py:90

    def prepare_data(self):
        # Called once on rank-0 only. Good for downloading data.
        # Do NOT assign instance state here (not called on all ranks).
        pass  # not implemented in our module

    def setup(self, stage: str | None = None):
        # Called on every GPU/worker rank.
        # Good for loading stats that every worker needs.
        stats_path = self._root / "processed/stats.json"
        if stats_path.exists():
            with open(stats_path) as f:
                self._stats = json.load(f)     # dataset.py:108-110

    def train_dataloader(self):
        return DataLoader(self._make_dataset("train"), ...)

    def val_dataloader(self):
        return DataLoader(self._make_dataset("val"), ...)
```

### Why `setup()` not `__init__()`

`setup()` is called after Lightning decides the rank. In multi-GPU training, each process calls `setup()` independently — so each worker can set its own `_stats` without communication. If you loaded stats in `__init__()`, you'd get a race condition when Lightning spawns workers.

In single-GPU training (our default), it doesn't matter, but writing `setup()` correctly makes the code DDP-safe for free.

### The `stage` parameter

```python
def setup(self, stage: str | None = None) -> None:
    # stage == "fit"     → called before trainer.fit()
    # stage == "test"    → called before trainer.test()
    # stage == "predict" → called before trainer.predict()
    # stage == None      → called for all stages
```

Our implementation ignores `stage` and loads stats unconditionally. This is fine for a single experiment but in a test suite you'd want to guard to avoid loading unnecessary files.

### `collate_fn` — padding variable-length sequences

```python
def collate_fn(batch: list[dict]) -> dict[str, torch.Tensor]:  # dataset.py:70
    def pad_sequence(seqs, pad_val=0.0):
        max_len = max(s.shape[-1] for s in seqs)
        padded = torch.full((len(seqs), *seqs[0].shape[:-1], max_len), pad_val)
        for i, s in enumerate(seqs):
            padded[i, ..., : s.shape[-1]] = s
        return padded
```

Each utterance in a batch has a different mel length. `pad_sequence` creates a fixed-size tensor filled with zeros, then writes each utterance into the correct slice. The `...` (Ellipsis) handles both 1D (`phoneme_ids`) and 2D (`mel`) sequences with the same function.

```
mel shapes in a batch of 3:
  utterance 0: (80, 120)
  utterance 1: (80, 95)
  utterance 2: (80, 150)

After pad_sequence:
  output: (3, 80, 150)  — padded to max length
```

The DataLoader receives the `collate_fn` as a parameter:

```python
DataLoader(
    dataset,
    batch_size=16,
    shuffle=True,
    collate_fn=collate_fn,   # dataset.py:127
    pin_memory=True,         # pre-pin CPU tensors for faster GPU transfer
    drop_last=True,          # discard the last incomplete batch during training
)
```

`pin_memory=True` allocates tensors in page-locked (pinned) memory, enabling asynchronous CPU→GPU transfer. Only beneficial when you have a GPU — no effect otherwise.

---

## Module 5 — The Trainer and Its Flags

### Every flag in `run_training()`

```python
trainer = L.Trainer(             # trainer.py:67
    max_epochs=1000,
    gradient_clip_val=1.0,
    accumulate_grad_batches=1,
    precision="bf16-mixed",
    val_check_interval=1.0,
    log_every_n_steps=50,
    logger=mlflow_logger,
    callbacks=[checkpoint_cb, early_stop_cb, MLflowMetricsCallback()],
    enable_progress_bar=True,
)
```

| Flag | Value | What it does |
|---|---|---|
| `max_epochs` | 1000 | Stop after 1000 full passes through the training data |
| `gradient_clip_val` | 1.0 | `clip_grad_norm_(params, 1.0)` before every optimizer step |
| `accumulate_grad_batches` | 1 | No accumulation (effective batch = batch_size × 1 = 16) |
| `precision` | `"bf16-mixed"` | Use bfloat16 for forward pass, float32 for optimizer |
| `val_check_interval` | 1.0 | Validate once per epoch (1.0 = 100% of training set) |
| `log_every_n_steps` | 50 | Write metrics to logger every 50 steps |
| `logger` | `MLFlowLogger` | The logger backend — receives all `self.log()` calls |
| `callbacks` | list | Called at lifecycle hooks (see Module 6) |
| `enable_progress_bar` | True | Show tqdm bar |

### `gradient_clip_val`

This is the same `torch.nn.utils.clip_grad_norm_` from Course 4, but the Trainer calls it automatically. The value 1.0 is set in `params.yaml:grad_clip_val`. If any gradient has norm > 1.0, all gradients are scaled down proportionally so the total norm equals 1.0.

### `accumulate_grad_batches`

When set to N, Lightning calls `optimizer.step()` only every N batches, accumulating gradients in between. Effective batch size = `batch_size × N`. Used when a large batch doesn't fit in GPU memory:

```
accumulate_grad_batches=4, batch_size=16 → effective batch = 64
```

Currently set to 1 (no accumulation), so this has no effect — but it's ready to be increased if training on a smaller GPU.

### `val_check_interval`

Can be a float (fraction of epoch) or int (number of steps):

```python
val_check_interval=1.0    # validate every epoch (after 100% of training batches)
val_check_interval=0.5    # validate twice per epoch
val_check_interval=500    # validate every 500 steps
```

### Mixed precision: `"bf16-mixed"`

In mixed-precision training:
- **Forward pass:** tensors are cast to bfloat16 (16-bit, same exponent range as float32 but less precision)
- **Backward pass:** gradients computed in bfloat16
- **Optimizer step:** master weights stored in float32 (no loss of precision in updates)

bfloat16 vs float16:
- float16: 5-bit exponent, 10-bit mantissa → can overflow/underflow, needs loss scaling
- bfloat16: 8-bit exponent (same as float32), 7-bit mantissa → no overflow risk, no loss scaling needed

This is why the project uses `"bf16-mixed"` over `"16-mixed"` — simpler and stable on modern GPUs (A100/3090/4090).

---

## Module 6 — Callbacks

### The callback hook system

A `Callback` is a plain Python class with methods named after lifecycle events. The Trainer calls every registered callback's corresponding method at the right moment:

```
trainer.fit() lifecycle (simplified):

  on_train_start()
  for epoch in range(max_epochs):
      on_train_epoch_start()
      for batch in train_loader:
          on_train_batch_start()
          training_step()                ← your LightningModule
          on_train_batch_end()
      on_train_epoch_end()               ← MLflowMetricsCallback hook
      
      on_validation_epoch_start()
      for batch in val_loader:
          validation_step()              ← your LightningModule
      on_validation_epoch_end()          ← AudioLoggerCallback hook
                                         ← ModelCheckpoint hook
                                         ← EarlyStopping hook
  on_train_end()
```

### `AudioLoggerCallback`

```python
class AudioLoggerCallback(L.Callback):         # callbacks.py:20

    def on_validation_epoch_end(self, trainer, pl_module):
        if not mlflow.active_run():
            return
        pl_module.eval()                       # switch to eval mode
        with torch.no_grad():
            for i, text in enumerate(self._val_texts):
                phoneme_ids = ...
                mel = pl_module.model.synthesize(phoneme_ids, phoneme_lengths, n_steps=10)
                np.save(f.name, mel.cpu().numpy())
                mlflow.log_artifact(f.name, ...)
        pl_module.train()                      # restore training mode
```

This callback manually calls `pl_module.eval()` and `pl_module.train()` because:
1. The Trainer called `pl_module.eval()` before the validation loop
2. `on_validation_epoch_end` runs after that loop — the module is still in eval mode
3. The callback needs to do inference (correct), then must restore train mode for the next epoch

This is the same eval/no_grad/train pattern from Course 4's `AudioLoggerCallback` module.

### `MLflowMetricsCallback`

```python
class MLflowMetricsCallback(L.Callback):      # callbacks.py:46

    def on_train_epoch_end(self, trainer, pl_module):
        if mlflow.active_run():
            metrics = {k: float(v) for k, v in trainer.callback_metrics.items()}
            mlflow.log_metrics(metrics, step=trainer.global_step)
```

`trainer.callback_metrics` is a dict of all metrics logged via `self.log()` during the epoch. The callback reads them and pushes them to MLflow manually — this supplements the `MLFlowLogger` which logs automatically but sometimes loses metrics in edge cases (e.g. when the run was created externally).

### `ModelCheckpoint`

```python
checkpoint_cb = ModelCheckpoint(           # trainer.py:54
    dirpath=str(checkpoint_dir),
    filename="best_model",
    monitor="val/loss",
    mode="min",
    save_top_k=3,
)
```

| Parameter | Value | Meaning |
|---|---|---|
| `monitor` | `"val/loss"` | Watch this metric |
| `mode` | `"min"` | Lower is better |
| `save_top_k` | 3 | Keep the 3 best checkpoints, delete the rest |
| `filename` | `"best_model"` | Checkpoint filename (Lightning appends epoch/step/metric) |

`checkpoint_cb.best_model_path` gives the path to the single best checkpoint after training ends — used in `run_training()` to log the model to MLflow.

### `EarlyStopping`

```python
early_stop_cb = EarlyStopping(            # trainer.py:61
    monitor="val/loss",
    patience=50,
    mode="min",
)
```

If `val/loss` does not improve for 50 consecutive validation checks, training stops. With `val_check_interval=1.0`, one check = one epoch, so this means 50 epochs without improvement.

---

## Module 7 — Checkpoint Saving and Resuming

### What a `.ckpt` file contains

A Lightning checkpoint is a plain `torch.save()` dictionary with these keys:

```python
{
    "epoch": 42,
    "global_step": 15000,
    "state_dict": {...},          # model weights
    "optimizer_states": [...],    # optimizer momentum, etc.
    "lr_schedulers": [...],       # scheduler state
    "hyper_parameters": {...},    # from save_hyperparameters()
    "pytorch-lightning_version": "...",
}
```

The `state_dict` is identical to what `torch.save(model.state_dict(), ...)` would save. The extra keys let Lightning resume training exactly where it left off.

### `auto_resume` classmethod

```python
@classmethod
def auto_resume(cls, checkpoint_dir, model_cfg, training_cfg):  # model.py:160
    checkpoints = sorted(Path(checkpoint_dir).glob("*.ckpt"))
    if checkpoints:
        latest = checkpoints[-1]
        return cls.load_from_checkpoint(latest, model_cfg=model_cfg, training_cfg=training_cfg)
    return cls(model_cfg=model_cfg, training_cfg=training_cfg)
```

This pattern:
1. Looks for `.ckpt` files in the checkpoint directory
2. Takes the lexicographically last one (alphabetical sort → last saved)
3. If found: deserializes weights + optimizer + scheduler from checkpoint
4. If not found: creates a fresh model

`load_from_checkpoint` reads `hyper_parameters` from the file, then calls `__init__` with those parameters, then loads the `state_dict`. The explicit `model_cfg=...` overrides the stored hparams — useful when resuming with slightly different config.

### The `trainer.fit()` resume path

When `trainer.fit(pl_model, datamodule=data_module)` is called and `pl_model` was loaded from a checkpoint, Lightning also restores `epoch`, `global_step`, and the LR scheduler state — so training continues from the exact batch it stopped at, not from epoch 0.

---

## Module 8 — MLFlowLogger Integration

### Creating the logger

```python
mlflow_logger = MLFlowLogger(              # trainer.py:29
    experiment_name=cfg.experiment.name,
    run_name=cfg.experiment.run_name,
    tracking_uri=cfg.mlflow.tracking_uri,
    tags=dict(cfg.experiment.tags),
)
run_id = mlflow_logger.run_id             # trainer.py:79
```

The `MLFlowLogger` wraps an MLflow run. When you pass it to `L.Trainer(logger=...)`, every `self.log()` call in your `LightningModule` is forwarded to this logger. The `run_id` is accessible immediately after construction — before `trainer.fit()` is called — so it can be passed to `mlflow.start_run()`.

### Nesting the MLflow context manager

```python
with mlflow.start_run(run_id=run_id):          # trainer.py:82
    mlflow.set_tag("dvc_commit", dvc_commit)
    trainer.fit(pl_model, datamodule=data_module)
    mlflow.pytorch.log_model(pl_model, "matcha_tts")
```

By passing `run_id=run_id`, `mlflow.start_run()` reuses the run already opened by `MLFlowLogger` — no duplicate run is created. Everything logged inside the `with` block (tags, artifacts, the model itself) goes into the same MLflow run as the metrics from `self.log()`.

---

## Module 9 — The Full Training Call Graph

Walk through `trainer.fit(pl_model, datamodule=data_module)` step by step:

```
1. Trainer.__init__() (already done)
   → stores callbacks, logger, flags

2. trainer.fit()
   ├─ data_module.setup("fit")              → loads stats.json
   ├─ train_loader = data_module.train_dataloader()
   ├─ val_loader   = data_module.val_dataloader()
   ├─ [configure_optimizers() called once]  → AdamW + SequentialLR
   │
   └─ Training loop (max_epochs=1000):
       ├─ [on_train_epoch_start]
       ├─ For each batch in train_loader:
       │   ├─ batch moved to device
       │   ├─ training_step(batch, batch_idx)   → returns loss
       │   ├─ loss.backward()
       │   ├─ clip_grad_norm_(params, 1.0)
       │   ├─ optimizer.step()
       │   ├─ scheduler.step()                  → interval="step"
       │   └─ optimizer.zero_grad()
       │
       ├─ [on_train_epoch_end]
       │   └─ MLflowMetricsCallback.on_train_epoch_end()
       │       → mlflow.log_metrics(trainer.callback_metrics)
       │
       ├─ [Validation loop — every val_check_interval]
       │   ├─ pl_module.eval()
       │   ├─ torch.no_grad()
       │   ├─ For each batch in val_loader:
       │   │   └─ validation_step(batch, batch_idx)   → logs val/loss
       │   ├─ [on_validation_epoch_end]
       │   │   ├─ AudioLoggerCallback.on_validation_epoch_end()
       │   │   │   → synthesize N samples → save to MLflow
       │   │   ├─ ModelCheckpoint.on_validation_epoch_end()
       │   │   │   → if val/loss improved → save .ckpt
       │   │   └─ EarlyStopping.on_validation_epoch_end()
       │   │       → if no improvement for 50 epochs → stop
       │   └─ pl_module.train()
       │
       └─ (repeat or stop)
```

This is the complete picture. Every callback, every hook, every automatic action in one place.

---

## Module 10 — Common Pitfalls

### Pitfall 1: Logging the same metric at different intervals

```python
# BAD — val/loss logged on_step=True during training, confuses dashboards
self.log("val/loss", loss, on_step=True)

# GOOD — validation metrics should be epoch-level only
self.log("val/loss", loss, on_step=False, on_epoch=True)
```

Lightning's default for `on_step` is `True` in `training_step` and `False` in `validation_step`. You rarely need to override these defaults.

### Pitfall 2: Returning a non-tensor from `training_step`

```python
# BAD — Lightning can't call .backward() on a dict
def training_step(self, batch, batch_idx):
    out = self.model(**batch)
    return out   # ← dict, not tensor

# GOOD
def training_step(self, batch, batch_idx):
    out = self.model(**batch)
    return out["loss"]   # ← scalar tensor
```

### Pitfall 3: Modifying model state in `__init__` before `super().__init__()`

```python
# BAD — hparams not yet initialized when self.model is assigned
class MyModule(L.LightningModule):
    def __init__(self, cfg):
        self.model = MyModel(cfg)   # ← error: __dict__ not set up
        super().__init__()

# GOOD
class MyModule(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.model = MyModel(cfg)
```

### Pitfall 4: Calling `optimizer.zero_grad()` manually

Lightning already calls it. If you add it in `training_step`, you'll zero the gradients before they're used, breaking accumulation.

### Pitfall 5: `monitor` metric not logged

If `ModelCheckpoint(monitor="val/loss")` but you never call `self.log("val/loss", ...)`, Lightning raises an error at the end of the first validation epoch. The metric name in `monitor` must exactly match what you pass to `self.log()`.

---

## Module 11 — Mini Practice Project: `lightkit`

An independent mini-project that reproduces the full Lightning pattern — `LightningModule`, `LightningDataModule`, callbacks, `ModelCheckpoint`, `EarlyStopping` — on a tiny regression problem. No TTS pipeline code.

**Goal:** Train a 2-layer MLP to fit a noisy sine curve. Log metrics, save the best checkpoint, and stop early if it stops improving.

### Project structure

```
lightkit/
├── model.py        # SineFitter (LightningModule)
├── data.py         # SineDataModule (LightningDataModule)
├── callbacks.py    # PlotCallback
└── train.py        # main()
```

### `data.py`

```python
import torch
from torch.utils.data import DataLoader, Dataset, random_split
import lightning as L


class SineDataset(Dataset):
    def __init__(self, n=2000, noise=0.05):
        x = torch.linspace(0, 4 * torch.pi, n).unsqueeze(1)   # (N, 1)
        y = torch.sin(x) + torch.randn_like(x) * noise         # (N, 1)
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class SineDataModule(L.LightningDataModule):
    def __init__(self, batch_size=64):
        super().__init__()
        self.batch_size = batch_size
        self.train_ds = None
        self.val_ds = None

    def setup(self, stage=None):
        full = SineDataset()
        n_val = int(len(full) * 0.1)
        self.train_ds, self.val_ds = random_split(full, [len(full) - n_val, n_val])

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size)
```

### `model.py`

```python
import torch
import torch.nn as nn
import lightning as L


class SineFitter(L.LightningModule):
    def __init__(self, hidden=64, lr=1e-3, warmup_steps=100):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = nn.functional.mse_loss(self(x), y)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss = nn.functional.mse_loss(self(x), y)
        self.log("val/loss", loss, prog_bar=True)

    def configure_optimizers(self):
        from torch.optim import AdamW
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

        opt = AdamW(self.parameters(), lr=self.hparams.lr)
        warmup = LinearLR(opt, start_factor=0.1, total_iters=self.hparams.warmup_steps)
        cosine = CosineAnnealingLR(opt, T_max=200)
        scheduler = SequentialLR(opt, [warmup, cosine], milestones=[self.hparams.warmup_steps])
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
```

### `callbacks.py`

```python
import lightning as L


class PlotCallback(L.Callback):
    """Print a mini ASCII plot of val loss after each epoch."""

    def __init__(self, width=40):
        self._losses = []
        self._width = width

    def on_validation_epoch_end(self, trainer, pl_module):
        val_loss = trainer.callback_metrics.get("val/loss")
        if val_loss is None:
            return
        self._losses.append(float(val_loss))
        # Normalize to bar width
        if len(self._losses) < 2:
            return
        lo, hi = min(self._losses), max(self._losses)
        span = hi - lo or 1.0
        bar_len = int((1.0 - (float(val_loss) - lo) / span) * self._width)
        bar = "#" * bar_len + "." * (self._width - bar_len)
        print(f"  epoch {trainer.current_epoch:4d} | [{bar}] val/loss={float(val_loss):.5f}")
```

### `train.py`

```python
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from data import SineDataModule
from model import SineFitter
from callbacks import PlotCallback


def main():
    model = SineFitter(hidden=64, lr=1e-3, warmup_steps=100)
    dm = SineDataModule(batch_size=64)

    checkpoint_cb = ModelCheckpoint(
        dirpath="checkpoints/",
        filename="best",
        monitor="val/loss",
        mode="min",
        save_top_k=1,
    )
    early_stop_cb = EarlyStopping(
        monitor="val/loss",
        patience=20,
        mode="min",
    )

    trainer = L.Trainer(
        max_epochs=300,
        gradient_clip_val=1.0,
        val_check_interval=1.0,
        log_every_n_steps=10,
        callbacks=[checkpoint_cb, early_stop_cb, PlotCallback()],
        enable_progress_bar=True,
    )

    trainer.fit(model, datamodule=dm)
    print(f"\nBest checkpoint: {checkpoint_cb.best_model_path}")
    print(f"Best val/loss:  {checkpoint_cb.best_model_score:.5f}")

    # Resume from checkpoint
    resumed = SineFitter.load_from_checkpoint(checkpoint_cb.best_model_path)
    print(f"Resumed model lr: {resumed.hparams.lr}")


if __name__ == "__main__":
    main()
```

### Running it

```bash
pip install lightning
python train.py
```

Expected output (abbreviated):

```
  epoch    1 | [........................................] val/loss=0.15432
  epoch    2 | [####....................................] val/loss=0.08201
  epoch   10 | [####################....................] val/loss=0.02841
  epoch   50 | [#################################.......] val/loss=0.00287
  ...
  epoch  134 | [########################################] val/loss=0.00251
Epoch 00154: early stopping triggered (patience=20)

Best checkpoint: checkpoints/best.ckpt
Best val/loss:  0.00251
Resumed model lr: 0.001
```

### Exercises

1. **Change `val_check_interval` to `0.5`** — observe that `EarlyStopping` now checks twice per epoch (patience of 20 becomes 10 effective epochs).
2. **Remove `save_hyperparameters()`** — try `load_from_checkpoint()` — observe the error. Understand why the stored hparams are needed.
3. **Set `accumulate_grad_batches=4`** in the Trainer — print the effective batch size. Verify that the same number of `optimizer.step()` calls happen (check by printing inside a custom callback's `on_before_optimizer_step`).
4. **Add `precision="bf16-mixed"`** — time 100 steps with and without. On CPU this may be slower (no BF16 hardware); on a CUDA GPU it should be faster.
5. **Implement a second callback** `LRMonitorCallback` that logs `trainer.optimizers[0].param_groups[0]["lr"]` at every step to a CSV file. Plot it to verify the warmup→cosine shape.

---

## Key Takeaways

- `LightningModule` holds model + training logic; the Trainer holds infrastructure. You never write `loss.backward()` or `batch.to(device)`.
- `configure_optimizers` returns a dict with `"lr_scheduler": {"interval": "step"}` to step the scheduler per batch, not per epoch.
- `LightningDataModule.setup()` is called per-rank, making multi-GPU DDP correct by default; `prepare_data()` is rank-0 only.
- `collate_fn` is responsible for padding variable-length sequences in a batch to the max length.
- Callbacks hook into named lifecycle events (`on_validation_epoch_end`, `on_train_epoch_end`). The eval/no_grad/train guard pattern is the callback author's responsibility.
- A checkpoint stores weights + optimizer + scheduler + hparams. `load_from_checkpoint` requires `save_hyperparameters()` to have been called.
- `MLFlowLogger` and `mlflow.start_run(run_id=...)` share the same run — no duplicates.