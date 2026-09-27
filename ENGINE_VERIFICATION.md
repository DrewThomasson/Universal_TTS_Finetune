# Adding and verifying a TTS engine

Use this checklist when adding an engine or changing its training and inference paths. Update the results table after testing.

## Checklist

- [ ] Use the official APIs and compatible dependencies. Document model assets, licenses and hardware requirements; isolate optional runtimes when dependencies conflict.
- [ ] Connect dataset preparation, training, packaging, CLI and GUI. Show only supported controls; reject unsupported options clearly.
- [ ] Share language/checkpoint validation between CLI and GUI. Use the selected language's pretrained start. Warn about extra data/time for supported scratch training. Keep transcription and TTS language choices separate.
- [ ] Run actual training and confirm completed optimizer steps and losses. A dry run, step-zero log or saved file alone is not a training pass.
- [ ] Package the required files, reload that trained artifact and generate valid, non-silent audio through the CLI and GUI callback. Check GUI startup, controls and invalid selections.
- [ ] Test representative languages and pretrained variants. Record which were tested; a catalog entry does not prove a language was trained. Check regressions in existing engines.
- [ ] Build and run affected Docker images when their dependencies or files change. State which optional runtimes the image excludes. Offer E2A export only for accepted formats, matching `conf_models.py`.
- [ ] Record the exact code/upstream revisions, hardware, dataset/settings, completed steps, artifacts, inference results, failures/retries and limitations. Bound test processes and clean temporary data and cloud rentals.

## Recorded results

Updated 2026-09-27. **Pass** means the linked report establishes that check. **—** means unverified in this record, not a failure. Results belong to their tested commits; this is not a full-suite run at the current head.

| Engine | Training | Artifact | CLI inference | GUI inference callback | Training language | Tested SHA |
| --- | --- | --- | --- | --- | --- | --- |
| align_tts | 44 steps | Pass | — | — | en | 84a737b3 |
| delightful_tts | 81 steps | Pass | — | — | en | 31296e7f |
| fast_pitch | 45 steps | Pass | — | — | en | 31296e7f |
| fast_speech | 45 steps | Pass | — | — | en | 31296e7f |
| fastspeech2 | 45 steps | Pass | — | — | en | 31296e7f |
| glow_tts | 12 steps | Pass | — | — | it | 3ae2798c |
| neuralhmm_tts | 45 steps | Pass | — | — | en | 31296e7f |
| overflow | 45 steps | Pass | — | — | en | 31296e7f |
| speedy_speech | 45 steps | Pass | — | — | en | 31296e7f |
| tacotron2_capacitron | 45 steps | Pass | — | — | en | 31296e7f |
| tacotron2_dca | 12 steps | Pass | — | — | de | 3ae2798c |
| tacotron2_ddc | 45 steps | Pass | — | — | es | 3ae2798c |
| vits_tts | 45 steps | Pass | — | — | es | 3ae2798c |
| xtts_v1 | 45 steps | Pass | — | — | en | 31296e7f |
| xtts_v2 | 45 steps | Pass | — | — | en | 31296e7f |
| piper | 90 / 24 batches | ONNX | — | — | es_ES / es_MX | 3ae2798c |
| mms_vits (Fairseq) | 50 prior / 7 Aceh steps | Pass | Pass: ace | Pass: ace callback | eng / ace | e36d0a6 / ff78d51 |
| styletts2 | 2 CPU / 2 CUDA steps | Pass | Pass | Pass + browser | en | bcb16b7 / 0e14574; inference 80e9f7d |
| omnivoice | 2 CPU / 2 CUDA LoRA steps | Pass | Pass: en/es | Pass: es + browser | en | 0e14574; device-switch retest below |
| f5_tts | 11 / 3 steps | Pass | Pass: en/zh-cn | Pass: en/zh-cn | en / zh-cn | 3562404 |

* The earlier English MMS export reloaded and generated a WAV; the Aceh retest established separate CLI and GUI callback inference.

## Coverage and limitations

- CPU optimizer steps are recorded for Align TTS, Piper, StyleTTS2 and OmniVoice. The other 16 engines have not established CPU training passes here. Align completed four steps in a bounded 16 GiB / 4 CPU container; Piper completed 8/8 CPU batches (ONNX export was not rerun).
- The original 16 engines have historical training/packaging passes across several E2A component commits. Shared GUI/Docker checks do not establish actual GUI inference for each engine. MMS's earlier Docker runtime check was blocked.
- MMS/Fairseq VITS completed seven CUDA optimizer steps on eight synthetic Aceh clips, then packaged and reloaded the language-matched artifact for CLI and GUI callback inference. Its default mapped checkpoint download also passed after fixing the model-directory path. MMS ASR transcribed all eight clips without supplied transcripts; normalized character error was 11.4% on this tiny MMS-generated set, which is not a native-speaker benchmark. The Aceh test does not verify every MMS language, GUI browser operation, or Docker. The MMS checkpoint license is CC BY-NC 4.0.
- StyleTTS2 and OmniVoice passed automatic runtime setup, real CPU and CUDA training, packaging, CLI inference, GUI callbacks and browser audio playback. These were short synthetic English smoke datasets: StyleTTS2 used four training/two evaluation clips, batch two; OmniVoice used two training/one evaluation clips, batch one. Both completed one epoch and two optimizer steps on each device.
- StyleTTS2 was tested on a 12 GB RTX 3060 and CPU. English is the supported language. The CPU/12 GB guarded profile trains the initial acoustic stage and skips joint SLM adversarial training. A moved artifact reloaded through the CLI, browser and Docker; pinned WavLM assets and automatic runtime relocation were checked. Initial CPU device errors and an incomplete validation batch were fixed and retried; full failure tracebacks were retained locally.
- OmniVoice was trained on CPU and a rented 24 GB RTX 3090; English/Spanish inference passed. The 646 published language IDs were catalog-validated, not individually trained. Scratch training, local resume, reference voice cloning and E2A export are not exposed for its LoRA adapter. The cloud instance was destroyed after collecting results; an independent account check found zero instances.
- CPU and CUDA Docker images built and served the GUI. Optional runtimes install automatically on first use into the mounted model cache; they are not preinstalled in the base image. See the latest PR report for artifact portability and CPU-to-CUDA inference checks. A fresh Docker-created StyleTTS2 cache remained host-owned; native UFT rebuilt the container runtime automatically and generated audio without a permission repair. First-use downloads need network access and additional storage.
- F5-TTS v1 passed English and Chinese smoke training on a 12 GB RTX 3060. Online weights changed from the base; both languages passed CLI/GUI inference, artifact loading and GUI HTTP startup. The base Docker image built and started successfully; its optional F5 runtime was installed separately in a temporary GPU container for these tests.
- Language, transcription and export audits are separate from training/inference passes. Speech quality, long runs and every language/checkpoint variant remain unverified.

## Test reports

[Aceh MMS TTS/ASR](https://github.com/DrewThomasson/Universal_TTS_Finetune/pull/8#issuecomment-5858611429) · [Automatic runtime retest](https://github.com/DrewThomasson/Universal_TTS_Finetune/pull/8#issuecomment-5858350526) · [Original 16 engines](https://github.com/DrewThomasson/ebook2audiobook/pull/2107#issuecomment-5838669784) · [MMS](https://github.com/DrewThomasson/Universal_TTS_Finetune/pull/4#issuecomment-5842057557) · [StyleTTS2](https://github.com/DrewThomasson/Universal_TTS_Finetune/pull/6#issuecomment-5849246171) · [OmniVoice](https://github.com/DrewThomasson/Universal_TTS_Finetune/pull/6#issuecomment-5850234071) · [Language audit](https://github.com/DrewThomasson/Universal_TTS_Finetune/pull/4#issuecomment-5842806322) · [ASR/export](https://github.com/DrewThomasson/Universal_TTS_Finetune/pull/4#issuecomment-5842573562)

For each retest, preserve the prior evidence and record the new commit, settings, result and report link. Do not assume an old pass applies to new code.
