"""Fine-tune a Meta MMS/Fairseq VITS checkpoint using Coqui's VITS trainer."""

import json
import os
from pathlib import Path

import soundfile as sf
import torch
import torchaudio
from trainer import Trainer, TrainerArgs

from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import Vits, VitsAudioConfig
from TTS.utils.audio import AudioProcessor

MMS_CHECKPOINT = None
output_path = os.path.dirname(os.path.abspath(__file__))
dataset_path = Path(output_path).parent / "LJSpeech-1.1"

if not MMS_CHECKPOINT:
    raise ValueError("An MMS/Fairseq VITS starting checkpoint is required.")

base_config = json.loads((Path(MMS_CHECKPOINT).parent / "config.json").read_text(encoding="utf-8"))
sample_rate = base_config["data"]["sampling_rate"]

# Prepared UFT datasets use 22.05 kHz. The MMS generator expects its native
# sample rate, so resample the copied workspace dataset without touching input.
for wav_path in (dataset_path / "wavs").glob("*.wav"):
    audio, original_rate = sf.read(wav_path, dtype="float32", always_2d=True)
    if original_rate != sample_rate:
        waveform = torch.from_numpy(audio.T.copy())
        resampled = torchaudio.functional.resample(waveform, original_rate, sample_rate)
        sf.write(wav_path, resampled.T.numpy(), sample_rate)

dataset_config = BaseDatasetConfig(
    formatter="ljspeech", meta_file_train="metadata.csv", path=str(dataset_path)
)
audio_config = VitsAudioConfig(
    sample_rate=sample_rate,
    win_length=base_config["data"]["win_length"],
    hop_length=base_config["data"]["hop_length"],
    num_mels=base_config["data"]["n_mel_channels"],
    mel_fmin=base_config["data"]["mel_fmin"],
    mel_fmax=base_config["data"]["mel_fmax"],
)
config = VitsConfig(
    audio=audio_config,
    run_name="mms_vits_finetune",
    batch_size=8,
    eval_batch_size=1,
    num_loader_workers=0,
    num_eval_loader_workers=0,
    run_eval=True,
    test_delay_epochs=1000,
    epochs=100,
    max_audio_len=8 * sample_rate,
    use_phonemes=False,
    print_step=25,
    mixed_precision=False,
    output_path=output_path,
    datasets=[dataset_config],
    cudnn_benchmark=False,
)
sample_count = sum(1 for line in (dataset_path / "metadata.csv").read_text(encoding="utf-8").splitlines() if line.strip())
if sample_count < 2:
    raise ValueError("MMS training needs at least two clips for training and evaluation.")
config.eval_split_size = max(config.eval_split_size, 1.0 / sample_count)
ap = AudioProcessor.init_from_config(config)
train_samples, eval_samples = load_tts_samples(
    dataset_config,
    eval_split=True,
    eval_split_max_size=config.eval_split_max_size,
    eval_split_size=config.eval_split_size,
)
model = Vits(config, ap)
discriminator = model.disc
model.load_fairseq_checkpoint(config, Path(MMS_CHECKPOINT).parent, eval=False)
model.disc = discriminator
config = model.config

trainer = Trainer(
    TrainerArgs(),
    config,
    output_path,
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples,
)
trainer.fit()
