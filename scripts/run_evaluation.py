"""Evaluation pipeline CLI entry point — called by dvc.yaml evaluate stage."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    from tts_pipeline.evaluation.baseline import load_baseline
    from tts_pipeline.evaluation.metrics.intelligibility import compute_cer
    from tts_pipeline.evaluation.metrics.pesq_stoi import compute_pesq_stoi
    from tts_pipeline.evaluation.metrics.rtf import compute_rtf_stats
    from tts_pipeline.evaluation.metrics.utmos import compute_utmos
    from tts_pipeline.evaluation.reporter import generate_report
    from tts_pipeline.evaluation.synthesizer import synthesize_test_set
    from tts_pipeline.text.tokenizer import Tokenizer
    from tts_pipeline.utils.config import from_hydra
    from tts_pipeline.utils.registry import load_model

    pipeline_cfg = from_hydra(cfg)
    eval_cfg = pipeline_cfg.evaluation
    report_dir = Path("reports/latest")
    synth_dir = report_dir / "synthesized"

    tokenizer = Tokenizer.load(Path("artifacts/export/phoneme_vocab.json"))
    model_name = f"matcha-tts-{pipeline_cfg.data.language}"

    # Load candidate model
    model = load_model(model_name, stage="Staging")
    import torch
    from tts_pipeline.models.vocoder.hifigan import HiFiGANGenerator

    vocoder = HiFiGANGenerator()
    ckpt = torch.load("artifacts/checkpoints/hifigan_best.ckpt", map_location="cpu", weights_only=False)
    vocoder.load_state_dict(ckpt.get("state_dict", ckpt))

    # Check if synthesis already done (idempotent: skip if synth dir has files)
    if synth_dir.exists() and any(synth_dir.glob("*.wav")):
        print("Synthesis outputs found — recomputing metrics from existing audio")
        existing = list(synth_dir.glob("*.wav"))
        synthesis_results = {p.stem: {"wav_path": str(p), "rtf": 0.0, "duration_sec": 0.0} for p in existing}
    else:
        synthesis_results = synthesize_test_set(
            model=model,
            vocoder=vocoder,
            tokenizer=tokenizer,
            test_file=Path("data/splits/test.txt"),
            metadata_path=Path("data/metadata.csv"),
            output_dir=synth_dir,
            sample_rate=pipeline_cfg.data.sample_rate,
            n_ode_steps=eval_cfg.n_ode_steps_eval,
            batch_size=eval_cfg.batch_size,
        )

    # Compute metrics
    synth_wav_paths = [Path(v["wav_path"]) for v in synthesis_results.values()]

    import pandas as pd
    meta = pd.read_csv("data/metadata.csv").set_index("id")
    ref_wav_dir = Path("data/processed/audio_normalized")
    ref_paths = [ref_wav_dir / f"{uid}.wav" for uid in synthesis_results]
    references = [str(meta.loc[uid]["text"]) if uid in meta.index else "" for uid in synthesis_results]

    utmos_metrics = compute_utmos(synth_wav_paths)
    cer_metrics = compute_cer(synth_wav_paths, references, whisper_model_size=eval_cfg.whisper_model,
                               language=pipeline_cfg.data.language)
    sig_metrics = compute_pesq_stoi(synth_wav_paths, ref_paths)
    rtf_metrics = compute_rtf_stats([v.get("rtf", 0.0) for v in synthesis_results.values()])  # type: ignore[arg-type]

    all_metrics = {**utmos_metrics, **cer_metrics, **sig_metrics, **rtf_metrics}

    # Load baseline for delta comparison
    baseline_model, baseline_version = load_baseline(model_name)
    baseline_metrics = None
    if baseline_model is not None:
        baseline_report = report_dir / "baseline_metrics.json"
        if baseline_report.exists():
            with open(baseline_report) as f:
                baseline_metrics = json.load(f)

    report = generate_report(
        metrics=all_metrics,
        baseline_metrics=baseline_metrics,
        synthesis_results=synthesis_results,
        absolute_floor=eval_cfg.absolute_floor,
        relative_delta=eval_cfg.relative_delta,
        report_dir=report_dir,
        model_info={"model_name": model_name, "stage": "Staging"},
    )

    if not report["pass"]:
        print("Evaluation FAILED — see reports/latest/evaluation_report.json")
        sys.exit(1)

    print("Evaluation PASSED — promoting to Shadow stage")
    from tts_pipeline.utils.registry import get_latest_version, transition_stage
    version = get_latest_version(model_name, stage="Staging")
    if version:
        transition_stage(model_name, version, stage="Shadow")


if __name__ == "__main__":
    main()
