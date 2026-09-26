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

Updated 2026-09-26. **Pass** means the linked report establishes that check. **—** means unverified in this record, not a failure. Results belong to their tested commits; this is not a full-suite run at the current head.

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
| mms_vits (Fairseq) | 50 steps | Pass | —* | — | eng | e36d0a6 |
| styletts2 | 10 steps | Pass | Pass | Pass | en | db67c2a |
| omnivoice | 10 LoRA steps | Pass | Pass: en/es | Pass: es | en | 550ec3a |
| f5_tts | 11 / 3 steps | Pass | Pass: en/zh-cn | Pass: en/zh-cn | en / zh-cn | 3562404 |

* MMS's exported model reloaded and generated a valid WAV; a separate CLI inference check is not established by its report.

## Coverage and limitations

- The original 16 engines have historical training/packaging passes across several E2A component commits. Their shared GUI/Docker checks do not establish actual GUI inference for each engine.
- MMS's exported model reloaded and generated audio; Docker runtime was blocked on the tested host.
- StyleTTS2 and OmniVoice passed GUI HTTP startup and real inference callbacks. A complete browser walkthrough is not recorded. Their optional runtimes are not bundled into or runtime-tested in the base Docker image.
- StyleTTS2 was tested in English on a 12 GB RTX 3060. Its guarded profile skips joint SLM adversarial training.
- OmniVoice was trained in English on a 24 GB RTX 3090; English/Spanish inference passed. The 646 published language IDs were catalog-validated, not individually trained. Scratch training, local resume, reference voice cloning and E2A export are not exposed for its LoRA adapter.
- F5-TTS v1 passed English and Chinese smoke training on a 12 GB RTX 3060. Online weights changed from the base; both languages passed CLI/GUI inference, artifact loading and GUI HTTP startup. The base Docker image built and started successfully; its optional F5 runtime was installed separately in a temporary GPU container for these tests.
- Language, transcription and export audits are separate from training/inference passes. Speech quality, long runs and every language/checkpoint variant remain unverified.

## Test reports

[Original 16 engines](https://github.com/DrewThomasson/ebook2audiobook/pull/2107#issuecomment-5838669784) · [MMS](https://github.com/DrewThomasson/Universal_TTS_Finetune/pull/4#issuecomment-5842057557) · [StyleTTS2](https://github.com/DrewThomasson/Universal_TTS_Finetune/pull/6#issuecomment-5849246171) · [OmniVoice](https://github.com/DrewThomasson/Universal_TTS_Finetune/pull/6#issuecomment-5850234071) · [Language audit](https://github.com/DrewThomasson/Universal_TTS_Finetune/pull/4#issuecomment-5842806322) · [ASR/export](https://github.com/DrewThomasson/Universal_TTS_Finetune/pull/4#issuecomment-5842573562)

For each retest, preserve the prior evidence and record the new commit, settings, result and report link. Do not assume an old pass applies to new code.
