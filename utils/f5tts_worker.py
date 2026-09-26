"""Optional F5-TTS runtime entry point; uses upstream model and Trainer APIs."""
from __future__ import annotations

import csv
import json
import math
import os
import shutil
import sys
from importlib.resources import files
from pathlib import Path


def train(config):
    import soundfile as sf
    import torch
    import torchaudio
    from cached_path import cached_path
    from f5_tts.model import CFM, DiT, Trainer
    from f5_tts.model.dataset import CustomDataset
    from f5_tts.model.utils import convert_char_to_pinyin, get_tokenizer

    root = Path(config['training_root'])
    ready = root / 'ready'
    ready.mkdir(exist_ok=True)
    vocab = ready / 'vocab.txt'
    shutil.copy2(files('f5_tts').joinpath('infer/examples/vocab.txt'), vocab)
    mapping, size = get_tokenizer(str(vocab), 'custom')
    rows = []
    for row in csv.reader((Path(config['dataset_dir']) / 'metadata_train.csv').open(encoding='utf-8'), delimiter='|'):
        if len(row) < 2:
            continue
        source = Path(config['dataset_dir']) / 'wavs' / (row[0] if Path(row[0]).suffix else row[0] + '.wav')
        audio, rate = sf.read(source, dtype='float32', always_2d=True)
        duration = len(audio) / rate
        if not 0.3 <= duration <= min(30, config['max_audio_seconds']):
            continue
        text = convert_char_to_pinyin([row[2] if len(row) > 2 else row[1]])[0]
        unknown = set(text) - set(mapping)
        if unknown:
            raise ValueError(f'F5-TTS base vocabulary cannot encode {unknown!r} in {source.name}.')
        destination = root / 'audio' / source.name
        destination.parent.mkdir(exist_ok=True)
        tensor = torch.from_numpy(audio.mean(axis=1)).unsqueeze(0)
        if rate != 24000:
            tensor = torchaudio.functional.resample(tensor, rate, 24000)
        sf.write(destination, tensor.squeeze(0).numpy(), 24000)
        rows.append(dict(audio_path=str(destination), text=text, duration=duration))
    updates = math.ceil(math.ceil(len(rows) / config['batch_size']) / config['grad_accum']) * config['epochs']
    if updates < 2:
        raise ValueError('F5-TTS needs at least two optimizer updates for its warmup/decay schedule.')
    if not torch.cuda.is_available():
        raise ValueError('This F5-TTS training profile requires CUDA.')
    free, _ = torch.cuda.mem_get_info()
    if free < 9 * 1024**3:
        raise ValueError('F5-TTS training needs at least 9 GiB free GPU memory for this profile; close other GPU tasks or use a larger GPU.')
    if shutil.disk_usage(root).free < 12 * 1024**3:
        raise ValueError('F5-TTS training needs at least 12 GiB free disk space for base and training checkpoints.')
    checkpoint_dir = root / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)
    base = config.get('restore_path') or str(cached_path('hf://SWivid/F5-TTS/F5TTS_v1_Base/model_1250000.safetensors', cache_dir=str(Path(os.environ.get('HF_HOME', root)) / 'f5_downloads')))
    shutil.copy2(base, checkpoint_dir / ('pretrained_' + Path(base).name))
    model = CFM(transformer=DiT(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4, text_num_embeds=size, mel_dim=100),
                mel_spec_kwargs=dict(n_fft=1024, hop_length=256, win_length=1024, n_mel_channels=100, target_sample_rate=24000, mel_spec_type='vocos'), vocab_char_map=mapping)
    trainer = Trainer(model, config['epochs'], 1e-5, num_warmup_updates=1,
                      save_per_updates=updates, keep_last_n_checkpoints=0, last_per_updates=updates,
                      checkpoint_path=str(checkpoint_dir), batch_size_per_gpu=config['batch_size'], batch_size_type='sample',
                      grad_accumulation_steps=config['grad_accum'], logger=None, log_samples=False,
                      bnb_optimizer=True, accelerate_kwargs={'mixed_precision': 'fp16'})
    trainer.train(CustomDataset(rows), num_workers=1, resumable_with_seed=666)
    checkpoint = torch.load(checkpoint_dir / 'model_last.pt', map_location='cpu', weights_only=True)
    completed = int(checkpoint['update'])
    if completed < updates:
        raise RuntimeError(f'F5-TTS completed {completed} updates; expected {updates}.')
    if any(not torch.isfinite(weight).all() for weight in checkpoint['model_state_dict'].values() if weight.is_floating_point()):
        raise RuntimeError('F5-TTS produced non-finite trained weights.')
    # Store online weights: early EMA checkpoints can still resemble the base.
    torch.save({'model_state_dict': checkpoint['model_state_dict']}, ready / 'model.pt')
    reference = rows[0]
    shutil.copy2(reference['audio_path'], ready / 'reference.wav')
    artifacts = dict(config, model_key='f5_tts', model_label='F5-TTS v1', family='f5_tts', checkpoint=str(ready / 'model.pt'),
                     vocab=str(vocab), reference_wav=str(ready / 'reference.wav'), reference_text=''.join(reference['text']),
                     trained_steps=completed, pretrained_model_id='SWivid/F5-TTS/F5TTS_v1_Base', use_ema=False)
    (ready / 'artifacts.json').write_text(json.dumps(artifacts, indent=2), encoding='utf-8')
    print(f'UFT_F5_COMPLETED_UPDATES={completed}', flush=True)


def infer(config):
    from f5_tts.model.utils import convert_char_to_pinyin, get_tokenizer
    mapping, _ = get_tokenizer(config['vocab'], 'custom')
    unknown = set(convert_char_to_pinyin([config['text']])[0]) - set(mapping)
    if unknown:
        raise ValueError(f'F5-TTS base vocabulary cannot encode {unknown!r}.')
    import soundfile as sf
    from f5_tts.api import F5TTS
    model = F5TTS(model='F5TTS_v1_Base', ckpt_file=config['checkpoint'], vocab_file=config['vocab'], use_ema=False)
    audio, rate, _ = model.infer(ref_file=config['reference_wav'], ref_text=config['reference_text'], gen_text=config['text'], seed=666)
    sf.write(config['output_file'], audio, rate)


if __name__ == '__main__':
    mode, path = sys.argv[1:]
    config = json.loads(Path(path).read_text(encoding='utf-8'))
    (train if mode == 'train' else infer)(config)
