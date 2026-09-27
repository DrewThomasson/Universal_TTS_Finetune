"""StyleTTS2 inference adapter based on the official LibriTTS demo."""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable


ProgressCallback = Callable[[str], None] | None
SAMPLE_RATE = 24000

# Inference_LibriTTS.ipynb's model loading and synthesis flow, adapted to take
# explicit paths and to allow CPU execution. Keep the model math aligned with
# the official notebook; generated noise is seeded for repeatable CLI/GUI runs.
_WORKER = r'''import json, random, sys, os
from pathlib import Path
args = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
repo = Path(args['repo'])
sys.path.insert(0, str(repo))
os.chdir(repo)
import numpy as np
import torch, yaml, librosa, torchaudio
from munch import Munch
from models import build_model, load_ASR_models, load_F0_models
from utils import recursive_munch
from text_utils import TextCleaner
from Utils.PLBERT.util import load_plbert
from Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule
import phonemizer
from nltk.tokenize import word_tokenize
import soundfile as sf

torch.manual_seed(args['seed']); random.seed(args['seed']); np.random.seed(args['seed'])
if args['device'] == 'cuda' and not torch.cuda.is_available():
    raise RuntimeError('CUDA was requested with UFT_STYLETTS2_INFER_DEVICE=cuda, but is unavailable.')
device = torch.device(args['device'])
with open(args['config'], encoding='utf-8') as f:
    config = yaml.safe_load(f)
for key, relative in {
    'ASR_config': 'Utils/ASR/config.yml',
    'ASR_path': 'Utils/ASR/epoch_00080.pth',
    'F0_path': 'Utils/JDC/bst.t7',
}.items():
    if not Path(config.get(key, '')).is_file():
        config[key] = str(repo / relative)
plbert_dir = Path(config.get('PLBERT_dir', ''))
if not plbert_dir.is_dir() or not list(plbert_dir.glob('step_*.t7')):
    config['PLBERT_dir'] = str(repo / 'Utils/PLBERT')
for key in ('ASR_config', 'ASR_path', 'F0_path'):
    if not Path(config[key]).is_file():
        raise FileNotFoundError('StyleTTS2 runtime asset not found: ' + config[key])
if not list(Path(config['PLBERT_dir']).glob('step_*.t7')):
    raise FileNotFoundError('StyleTTS2 PL-BERT checkpoint not found under ' + config['PLBERT_dir'])
text_aligner = load_ASR_models(config['ASR_path'], config['ASR_config'])
pitch_extractor = load_F0_models(config['F0_path'])
plbert = load_plbert(config['PLBERT_dir'])
model_params = recursive_munch(config['model_params'])
wavlm_dir = args.get('wavlm_dir')
configured_slm_model = str(model_params.slm.model)
configured_slm_path = Path(configured_slm_model).expanduser()
is_known_managed_wavlm = (configured_slm_path.is_absolute() and not configured_slm_path.exists()
                          and configured_slm_path.name == 'wavlm-base-plus'
                          and configured_slm_path.parent.name == 'styletts2')
if wavlm_dir and (configured_slm_model == 'microsoft/wavlm-base-plus' or is_known_managed_wavlm or
                  configured_slm_path == Path(wavlm_dir).expanduser()):
    model_params.slm.model = wavlm_dir
model = build_model(model_params, text_aligner, pitch_extractor, plbert)
for key in model:
    model[key].eval().to(device)
checkpoint = torch.load(args['checkpoint'], map_location='cpu', weights_only=False)
params = checkpoint.get('net', checkpoint)
for key in model:
    if key not in params: continue
    state = params[key]
    try:
        model[key].load_state_dict(state)
    except RuntimeError:
        if state and all(name.startswith('module.') for name in state):
            model[key].load_state_dict({name[7:]: value for name, value in state.items()}, strict=False)
        else:
            raise
for key in model: model[key].eval()
sampler = DiffusionSampler(model.diffusion.diffusion, sampler=ADPM2Sampler(),
    sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0), clamp=False)
cleaner = TextCleaner()
phonemizer_backend = phonemizer.backend.EspeakBackend(language='en-us', preserve_punctuation=True, with_stress=True)
to_mel = torchaudio.transforms.MelSpectrogram(n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
def compute_style(path):
    wave, _ = librosa.load(path, sr=24000)
    wave, _ = librosa.effects.trim(wave, top_db=30)
    mel = (torch.log(1e-5 + to_mel(torch.from_numpy(wave).float()).unsqueeze(0)) + 4) / 4
    with torch.no_grad():
        value = mel.unsqueeze(1).to(device)
        return torch.cat([model.style_encoder(value), model.predictor_encoder(value)], dim=1)
def length_to_mask(lengths):
    mask = torch.arange(lengths.max(), device=lengths.device).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
    return torch.gt(mask + 1, lengths.unsqueeze(1))
def infer(text, ref_s):
    phonemes = phonemizer_backend.phonemize([text.strip()])[0]
    phonemes = ' '.join(word_tokenize(phonemes, preserve_line=True))
    tokens = cleaner(phonemes); tokens.insert(0, 0)
    tokens = torch.LongTensor(tokens).to(device).unsqueeze(0)
    with torch.no_grad():
        lengths = torch.LongTensor([tokens.shape[-1]]).to(device)
        mask = length_to_mask(lengths)
        t_en = model.text_encoder(tokens, lengths, mask)
        bert_dur = model.bert(tokens, attention_mask=(~mask).int())
        d_en = model.bert_encoder(bert_dur).transpose(-1, -2)
        s_pred = sampler(noise=torch.randn((1,256), device=device).unsqueeze(1), embedding=bert_dur,
            embedding_scale=1, features=ref_s, num_steps=5).squeeze(1)
        s, ref = s_pred[:,128:], s_pred[:,:128]
        ref = .3 * ref + .7 * ref_s[:,:128]
        s = .7 * s + .3 * ref_s[:,128:]
        d = model.predictor.text_encoder(d_en, s, lengths, mask)
        x, _ = model.predictor.lstm(d)
        duration = torch.sigmoid(model.predictor.duration_proj(x)).sum(axis=-1)
        pred_dur = torch.round(duration.squeeze()).clamp(min=1)
        alignment = torch.zeros(int(lengths[0]), int(pred_dur.sum().item()), device=device)
        frame = 0
        for i in range(alignment.size(0)):
            count = int(pred_dur[i].item()); alignment[i, frame:frame+count] = 1; frame += count
        en = d.transpose(-1,-2) @ alignment.unsqueeze(0)
        asr = t_en @ alignment.unsqueeze(0)
        if model_params.decoder.type == 'hifigan':
            shifted = torch.zeros_like(en); shifted[:,:,0] = en[:,:,0]; shifted[:,:,1:] = en[:,:,:-1]; en = shifted
            shifted = torch.zeros_like(asr); shifted[:,:,0] = asr[:,:,0]; shifted[:,:,1:] = asr[:,:,:-1]; asr = shifted
        f0, noise = model.predictor.F0Ntrain(en, s)
        return model.decoder(asr, f0, noise, ref.squeeze().unsqueeze(0)).squeeze().cpu().numpy()[..., :-50]
style = compute_style(args['reference_wav'])
audio = infer(args['text'], style)
sf.write(args['output_file'], audio, 24000)
'''


def _required(path: Path, description: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"StyleTTS2 {description} not found: {path}")
    return path.resolve()


def _artifact_file(artifacts: dict[str, Any], key: str, filename: str, description: str) -> Path:
    """Resolve a run file, falling back to its standard location by artifacts.json."""
    value = artifacts.get(key)
    if value:
        saved = Path(value).expanduser()
        if saved.is_file():
            return saved.resolve()
    artifact_file = artifacts.get("artifacts_file")
    if artifact_file:
        relocated = Path(artifact_file).expanduser().resolve().parent / filename
        if relocated.is_file():
            return relocated
    raise FileNotFoundError(f"StyleTTS2 {description} not found: {value or filename}")


def _valid_repo(repo: Path) -> bool:
    return (repo / "models.py").is_file() and (repo / "Demo" / "Inference_LibriTTS.ipynb").is_file()


def _runnable_python(python: str | os.PathLike[str]) -> bool:
    try:
        result = subprocess.run([str(python), "-c", "pass"], capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def _pick_runtime(*, artifacts: dict[str, Any], explicit_repo: str | None,
                  explicit_python: str | None, cpu: bool,
                  progress: ProgressCallback = None) -> tuple[Path, str]:
    from setup_styletts2 import runtime_paths, is_ready, setup, wavlm_is_ready

    saved_repo = artifacts.get("styletts2_repo")
    saved_python = artifacts.get("python_executable")
    default_repo, _, default_python = runtime_paths()
    repo_override = explicit_repo or os.environ.get("UFT_STYLETTS2_REPO")
    python_override = explicit_python or os.environ.get("UFT_STYLETTS2_PYTHON")

    if repo_override:
        repo = Path(repo_override).expanduser().resolve()
        if not _valid_repo(repo):
            raise FileNotFoundError(f"StyleTTS2 source override is unavailable: {repo}")
    else:
        repo = next((Path(candidate).expanduser().resolve() for candidate in (saved_repo, default_repo)
                     if candidate and _valid_repo(Path(candidate).expanduser())), None)

    if python_override:
        if not _runnable_python(python_override):
            raise RuntimeError(f"StyleTTS2 Python override cannot run: {python_override}")
        python = str(Path(python_override).expanduser())
    else:
        python = next((str(Path(candidate).expanduser()) for candidate in (saved_python, default_python)
                       if candidate and _runnable_python(candidate)), "")

    if repo is None or not python:
        managed_repo, _, managed_python = setup(cpu=cpu, progress=progress)
        if repo is None:
            repo = Path(managed_repo).resolve()
        if not python:
            python = str(managed_python)
    if repo.resolve() == Path(default_repo).resolve():
        require_cuda_runtime = not cpu and not bool(python_override)
        runtime_ready = is_ready(cuda=require_cuda_runtime)
        if not runtime_ready or not wavlm_is_ready():
            managed_repo, _, managed_python = setup(cpu=cpu or bool(python_override), progress=progress)
            repo = Path(managed_repo).resolve()
            if not python_override:
                python = str(managed_python)
    if not _valid_repo(repo):
        raise FileNotFoundError(f"Expected official StyleTTS2 source checkout at {repo}.")
    if not _runnable_python(python):
        raise RuntimeError(f"StyleTTS2 Python interpreter cannot run: {python}")
    return repo, python


def synthesize_styletts2(*, artifacts: dict[str, Any], text: str, reference_wav: str | os.PathLike[str] | None,
                         output_file: str | os.PathLike[str], python_executable: str | None = None,
                         progress: ProgressCallback = None, device: str = "auto") -> dict[str, Any]:
    """Synthesize with a completed UFT StyleTTS2 run and explicit reference WAV."""
    if not text.strip():
        raise ValueError("Text is required for synthesis.")
    if not reference_wav:
        raise ValueError("StyleTTS2 inference requires a speaker reference WAV.")
    requested_device = os.environ.get("UFT_STYLETTS2_INFER_DEVICE", "cpu").lower() if device == "auto" else device
    if requested_device not in {"cpu", "cuda"}:
        raise ValueError("UFT_STYLETTS2_INFER_DEVICE must be 'cpu' or 'cuda'.")
    checkpoint = _artifact_file(artifacts, "checkpoint", "model.pth", "trained checkpoint")
    config_path = _artifact_file(artifacts, "config", "config_ft.yml", "fine-tuning config")
    reference = _required(Path(reference_wav).expanduser(), "speaker reference WAV")
    from setup_styletts2 import runtime_paths
    managed_repo, _, _ = runtime_paths()
    from setup_styletts2 import wavlm_path
    repo, python = _pick_runtime(
        artifacts=artifacts, explicit_repo=None, explicit_python=python_executable,
        cpu=requested_device == "cpu", progress=progress,
    )
    output = Path(output_file).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {"repo": str(repo), "checkpoint": str(checkpoint), "config": str(config_path),
               "reference_wav": str(reference), "output_file": str(output), "text": text, "seed": 0,
               "device": requested_device,
               "wavlm_dir": str(wavlm_path()) if repo.resolve() == managed_repo.resolve() else None}
    with tempfile.TemporaryDirectory(prefix="uft-styletts2-infer-", dir=output.parent) as temporary:
        payload_path = Path(temporary) / "request.json"
        worker_path = Path(temporary) / "worker.py"
        payload_path.write_text(json.dumps(payload), encoding="utf-8")
        worker_path.write_text(_WORKER, encoding="utf-8")
        if progress:
            progress(f"Running official StyleTTS2 LibriTTS inference on {requested_device}...")
        try:
            environment = {**os.environ, "PYTHONPATH": str(repo) + os.pathsep + os.environ.get("PYTHONPATH", ""),
                           "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
            from utils.styletts2_env import configure_styletts2_hf_home
            configure_styletts2_hf_home(environment, repo, managed_repo)
            result = subprocess.run([python, str(worker_path), str(payload_path)], cwd=repo,
                                    capture_output=True, text=True, timeout=900,
                                    env=environment)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("StyleTTS2 inference exceeded the 15 minute limit and was stopped.") from exc
        if result.returncode:
            detail = (result.stderr or result.stdout).strip()[-3000:]
            raise RuntimeError(f"StyleTTS2 inference failed in {python}. Check local dependencies/assets. {detail}")
    # The embedded script is also intentionally straightforward to validate in
    # mock-only environments without importing any StyleTTS2 dependencies.
    if not output.is_file():
        raise RuntimeError("StyleTTS2 inference completed without creating an audio file.")
    return {"model_key": "styletts2", "output_file": str(output), "speaker_wav": str(reference),
            "artifacts_file": artifacts.get("artifacts_file", "")}
