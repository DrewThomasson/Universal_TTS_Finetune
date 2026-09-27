# Engine options

Choose an engine and language in the GUI or CLI. The language and checkpoint lists show valid choices for that engine. If no starting checkpoint exists, UFT warns that training from scratch needs much more data. **Auto**, **CPU**, and **CUDA** are available in the device picker; the GUI and CLI show a RAM/VRAM estimate before training. Those estimates are for short clips and small batches.

Engines with a managed optional runtime install it into `models/` on first use and reuse it later, in local runs and Docker. You do not need to run a separate setup command for them. F5-TTS is the current exception and needs an external Python environment.

The dataset transcription language is separate from the training engine language. Automatic transcription uses Whisper or a published MMS ASR adapter; an exact transcript map skips ASR.

| Engine | Language / starting point | What you need to know |
| --- | --- | --- |
| StyleTTS2 | English; official LibriTTS base | Batch size 2 and a speaker reference WAV for inference. CPU and 12 GB GPU runs use the initial acoustic stage. [Official model terms](https://github.com/yl4579/StyleTTS2#pre-trained-models) require synthetic-speech disclosure unless you have cloning permission. No E2A ZIP. |
| OmniVoice | Published OmniVoice language catalog; official base | Linux; CUDA training needs at least 16 GB total VRAM and 12 GB free. CPU training needs at least 16 GB available RAM. [Weights](https://huggingface.co/k2-fsa/OmniVoice) are noncommercial; the [audio tokenizer has separate terms](https://huggingface.co/k2-fsa/OmniVoice/blob/main/audio_tokenizer/LICENSE). No E2A ZIP. |
| F5-TTS v1 | English or Chinese (`zh-cn`); official base | Needs a separate environment, at least 9 GB free VRAM for CUDA training, and 12 GB free disk. Its saved reference audio is used for inference. [Code is MIT; base weights are CC BY-NC 4.0](https://github.com/SWivid/F5-TTS). No E2A ZIP. |
| XTTS v1 / v2 | Published XTTS languages only (14 / 17) | Inference needs a speaker reference WAV. |
| MMS / Fairseq VITS | [Meta's MMS language catalog](https://dl.fbaipublicfiles.com/mms/tts/all-tts-languages.html); matching checkpoint required | Use the published code, such as `ace` or `eng`. Checkpoints are CC BY-NC 4.0. E2A ZIP is available. |
| Piper | Choose language, voice, and quality | A ready-to-speak ONNX voice is not a training checkpoint. The selected eSpeak voice must be installed. E2A ZIP is available. |
| Align TTS | English | Other languages are not currently exposed. |
| Other Coqui engines | Available phonemizer languages | Some languages have no pretrained start. A mapped checkpoint cannot expand its vocabulary; UFT reports unsupported dataset symbols. |

For F5-TTS, install PyTorch and `f5-tts==1.1.22` in a separate Python environment, then point UFT to its interpreter before starting the GUI or CLI:

```bash
export UFT_F5TTS_PYTHON=/path/to/f5-env/bin/python
```

The base Docker image does not include F5-TTS. Put its runtime in a custom image and set the same variable there. [ENGINE_VERIFICATION.md](ENGINE_VERIFICATION.md) records which training and inference paths have actually been tested; a listed language or device is not a pass for every combination.
