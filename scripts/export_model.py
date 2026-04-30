"""Export acoustic model + vocoder to ONNX with int8 quantization."""
from __future__ import annotations

import json
from pathlib import Path

import torch


def export_acoustic(model: torch.nn.Module, export_path: Path, n_mels: int = 80) -> None:
    model.eval()
    B, T_text = 1, 20
    dummy_ids = torch.zeros(B, T_text, dtype=torch.long)
    dummy_lengths = torch.tensor([T_text], dtype=torch.long)

    torch.onnx.export(
        model,
        (dummy_ids, dummy_lengths),
        str(export_path),
        opset_version=17,
        input_names=["phoneme_ids", "phoneme_lengths"],
        output_names=["mel"],
        dynamic_axes={
            "phoneme_ids": {0: "batch", 1: "text_len"},
            "phoneme_lengths": {0: "batch"},
            "mel": {0: "batch", 2: "mel_len"},
        },
        do_constant_folding=True,
    )
    print(f"Exported acoustic model → {export_path}")


def export_vocoder(model: torch.nn.Module, export_path: Path, n_mels: int = 80) -> None:
    model.eval()
    dummy_mel = torch.zeros(1, n_mels, 100)
    torch.onnx.export(
        model,
        dummy_mel,
        str(export_path),
        opset_version=17,
        input_names=["mel"],
        output_names=["audio"],
        dynamic_axes={
            "mel": {0: "batch", 2: "mel_len"},
            "audio": {0: "batch", 2: "audio_len"},
        },
        do_constant_folding=True,
    )
    print(f"Exported vocoder → {export_path}")


def quantize_onnx(onnx_path: Path) -> Path:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    q_path = onnx_path.with_suffix(".int8.onnx")
    quantize_dynamic(str(onnx_path), str(q_path), weight_type=QuantType.QInt8)
    print(f"Quantized → {q_path}")
    return q_path


def validate_onnx(onnx_path: Path) -> None:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    print(f"ONNX validation OK: {onnx_path.name} | inputs={[i.name for i in sess.get_inputs()]}")


def main() -> None:
    import subprocess
    from tts_pipeline.utils.config import load_params

    cfg = load_params()
    export_dir = Path("artifacts/export")
    export_dir.mkdir(parents=True, exist_ok=True)

    # Load acoustic model
    from tts_pipeline.models.matcha.model import MatchaTTSLightning
    from tts_pipeline.text.tokenizer import Tokenizer

    tokenizer = Tokenizer.load(export_dir / "phoneme_vocab.json")
    model_cfg = cfg.model.model_dump()
    model_cfg["vocab_size"] = tokenizer.vocab_size
    pl_model = MatchaTTSLightning.load_from_checkpoint(
        "artifacts/checkpoints/best_model.ckpt",
        model_cfg=model_cfg,
        training_cfg=cfg.training.model_dump(),
    )
    acoustic = pl_model.model

    # Export acoustic
    acoustic_raw = export_dir / "acoustic_raw.onnx"
    export_acoustic(acoustic, acoustic_raw)
    acoustic_final = export_dir / "acoustic.onnx"
    if cfg.export.quantize:
        q = quantize_onnx(acoustic_raw)
        q.rename(acoustic_final)
    else:
        acoustic_raw.rename(acoustic_final)
    validate_onnx(acoustic_final)

    # Load and export vocoder
    from tts_pipeline.models.vocoder.hifigan import HiFiGANGenerator

    vocoder = HiFiGANGenerator()
    ckpt = torch.load("artifacts/checkpoints/hifigan_best.ckpt", map_location="cpu", weights_only=False)
    vocoder.load_state_dict(ckpt.get("state_dict", ckpt))
    vocoder.remove_weight_norm()

    vocoder_raw = export_dir / "vocoder_raw.onnx"
    export_vocoder(vocoder, vocoder_raw)
    vocoder_final = export_dir / "vocoder.onnx"
    if cfg.export.quantize:
        q = quantize_onnx(vocoder_raw)
        q.rename(vocoder_final)
    else:
        vocoder_raw.rename(vocoder_final)
    validate_onnx(vocoder_final)

    # config.json
    config_out = {
        "sample_rate": cfg.data.sample_rate,
        "hop_length": cfg.data.hop_length,
        "n_mels": cfg.data.n_mels,
        "n_ode_steps": cfg.model.n_ode_steps,
        "language": cfg.data.language,
    }
    with open(export_dir / "config.json", "w") as f:
        json.dump(config_out, f, indent=2)

    # model_card.json
    dvc_commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    eval_report = {}
    eval_path = Path("reports/latest/evaluation_report.json")
    if eval_path.exists():
        with open(eval_path) as f:
            eval_report = json.load(f).get("metrics", {})

    with open(export_dir / "model_card.json", "w") as f:
        json.dump({
            "version": "0.1.0",
            "dvc_commit": dvc_commit,
            "language": cfg.data.language,
            "eval_metrics": eval_report,
        }, f, indent=2)

    print("Export complete →", export_dir)


if __name__ == "__main__":
    main()
