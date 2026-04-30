from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def _weights_init(m: nn.Module) -> None:
    if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
        nn.init.normal_(m.weight, 0.0, 0.01)


class ResBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, dilations: list[int] | None = None) -> None:
        super().__init__()
        dilations = dilations or [1, 3, 5]
        self.convs1 = nn.ModuleList([
            nn.utils.weight_norm(nn.Conv1d(channels, channels, kernel_size, dilation=d, padding=d * (kernel_size - 1) // 2))
            for d in dilations
        ])
        self.convs2 = nn.ModuleList([
            nn.utils.weight_norm(nn.Conv1d(channels, channels, kernel_size, dilation=1, padding=(kernel_size - 1) // 2))
            for _ in dilations
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, 0.1)
            xt = F.leaky_relu(c1(xt), 0.1)
            xt = c2(xt)
            x = x + xt
        return x


class HiFiGANGenerator(nn.Module):
    """HiFi-GAN V1 generator."""

    def __init__(
        self,
        n_mels: int = 80,
        upsample_rates: list[int] | None = None,
        upsample_kernel_sizes: list[int] | None = None,
        upsample_initial_channel: int = 512,
        resblock_kernel_sizes: list[int] | None = None,
        resblock_dilation_sizes: list[list[int]] | None = None,
    ) -> None:
        super().__init__()
        upsample_rates = upsample_rates or [8, 8, 2, 2]
        upsample_kernel_sizes = upsample_kernel_sizes or [16, 16, 4, 4]
        resblock_kernel_sizes = resblock_kernel_sizes or [3, 7, 11]
        resblock_dilation_sizes = resblock_dilation_sizes or [[1, 3, 5], [1, 3, 5], [1, 3, 5]]

        self.conv_pre = nn.utils.weight_norm(nn.Conv1d(n_mels, upsample_initial_channel, 7, padding=3))
        self.ups = nn.ModuleList()
        self.resblocks = nn.ModuleList()

        ch = upsample_initial_channel
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                nn.utils.weight_norm(
                    nn.ConvTranspose1d(ch, ch // 2, k, stride=u, padding=(k - u) // 2)
                )
            )
            ch //= 2
            for j, (rk, rd) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(ResBlock(ch, rk, rd))

        self.conv_post = nn.utils.weight_norm(nn.Conv1d(ch, 1, 7, padding=3))
        self.apply(_weights_init)
        self._n_resblocks = len(resblock_kernel_sizes)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        x = self.conv_pre(mel)
        for i, up in enumerate(self.ups):
            x = F.leaky_relu(x, 0.1)
            x = up(x)
            xs = None
            for j in range(self._n_resblocks):
                rb_out = self.resblocks[i * self._n_resblocks + j](x)
                xs = rb_out if xs is None else xs + rb_out
            x = xs / self._n_resblocks  # type: ignore[operator]
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        return torch.tanh(x)  # (B, 1, T_audio)

    def remove_weight_norm(self) -> None:
        for m in self.modules():
            if hasattr(m, "weight_g"):
                nn.utils.remove_weight_norm(m)


try:
    import lightning as L

    class HiFiGANLightning(L.LightningModule):
        def __init__(self, model_cfg: dict[str, Any], training_cfg: dict[str, Any]) -> None:
            super().__init__()
            self.save_hyperparameters()
            self.generator = HiFiGANGenerator(**{k: v for k, v in model_cfg.items() if k != "type"})
            self._training_cfg = training_cfg
            self.automatic_optimization = False

        def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
            opt_g, opt_d = self.optimizers()  # type: ignore[misc]
            mel, audio = batch["mel"], batch["audio"]

            # Generator step
            opt_g.zero_grad()
            fake_audio = self.generator(mel)
            g_loss = F.l1_loss(mel, self._mel_from_audio(fake_audio)) * 45
            self.manual_backward(g_loss)
            opt_g.step()
            self.log("train/g_loss", g_loss, prog_bar=True)

        def _mel_from_audio(self, audio: torch.Tensor) -> torch.Tensor:
            import torchaudio.transforms as T

            mel_tf = T.MelSpectrogram(sample_rate=22050, n_fft=1024, hop_length=256, n_mels=80).to(audio.device)
            return T.AmplitudeToDB()(mel_tf(audio.squeeze(1)))

        def configure_optimizers(self) -> Any:
            from torch.optim import AdamW

            opt_g = AdamW(self.generator.parameters(), lr=self._training_cfg.get("learning_rate", 2e-4))
            return [opt_g]

except ImportError:
    pass
