# Universal_TTS_Finetune
Attempt at making a universal script for fine-tuning a wide variety of tts models with a single command

# Confirmed working ljspeech dataset recipies from coqui tts

- [ ] align_tts  
- [ ] delightful_tts  
- [ ] fast_pitch  
- [ ] fast_speech  
- [ ] fastspeech2  
- [ ] glow_tts  
- [ ] hifigan  
- [ ] multiband_melgan  
- [ ] neuralhmm_tts  
- [ ] overflow  
- [ ] speedy_speech  
- [ ] tacotron2-Capacitron  
- [ ] tacotron2-DCA  
- [ ] tacotron2-DDC  
- [ ] univnet  
- [ ] vits_tts  
- [ ] wavegrad  
- [ ] wavernn  
- [ ] xtts_v1  
- [ ] xtts_v2  






## Extra Overkill for training models and such (All supported Coqio tts models and piper-tts in one easy command) 
- For info about this @DrewThomasson, he is currently working on the development of this, [work-in-progress-repo here](https://github.com/DrewThomasson/Universal_TTS_Finetune)
- [ ] Make a easy to use training gui for all coqio tts models in th ljspeech format training recipes [here from coqui tts](https://github.com/coqui-ai/TTS/tree/dev/recipes/ljspeech)
- More info:
- [Bark TTS training repo](https://github.com/anyvoiceai/Barkify)
- [Coqui Training docs](https://docs.coqui.ai/en/latest/training_a_model.html)
- [Coqui training Tutorial for beginners](https://docs.coqui.ai/en/latest/tutorial_for_nervous_beginners.html)
- [Formatting your dataset](https://docs.coqui.ai/en/latest/formatting_your_dataset.html)
- [ljspeech dataset generator(Uses whisper for transcription?)](https://github.com/davidmartinrius/speech-dataset-generator)
- [DeepFilterNet2 for high quality denoising training data](https://github.com/Rikorose/DeepFilterNet)
- [Styletts2 fine tuning](https://github.com/yl4579/StyleTTS2/discussions/144)
- [more styletts2 training](https://dagshub.com/blog/styletts2/)
- [styletts2 finetuning google colab](https://colab.research.google.com/github/yl4579/StyleTTS2/blob/main/Colab/StyleTTS2_Finetune_Demo.ipynb)
The GUI/headless should be as simple as:
- [On training piper-tts](https://github.com/rhasspy/piper/blob/master/TRAINING.md)
- Good news! They all appear to use LJSpeech FORMAT! Even the standard Piper-tts training code!
- Remember tho that the dataset generator thing repo you linked makes the names of the files in the dataset in the metadata.csvfile like this "wavs/1272-128104-0000.wav" But the LJspeech trainers all appear to want it like this "1272-128104-0000", So you should probs make the dataset generating function your making take that into account and have the final metatdata.csv file that it'll use to pass into the many trainer functions.
- I plan on giving this a gradio gui potentially,
#### Step 1:
- Select model to fine-tune
- Give input audio file/files
- Press process and create dataset
#### Step 2: (Once Process dataset has been completed)
- Change any training parameters/config from standard training config for that selected model
- Press Train model
#### Step 3: (once training is complete)
- Load trained model
- Test influencing the trained/fine-tuned model 
- Export model & zipped dataset OR Export condensed model & zipped dataset
#### That's it! Planned to be simple as that! 🎉

- Another Goal of this it to have it automatically be able to work on cpu and GPU, auto-selecting to whichever one is avalible


# Dataset creater checklist

- [x] modify formatter.py
- [ ] modify gradio gui to function with the new formatter.py and make all the output files into the datset folder
- [ ] metadata_shuf.csv
- [ ] metadata_train.csv
- [ ] metadata_val.csv
- [ ] metadata.csv
- [ ] rename the dataset folder to LJSpeech-1.1
- [ ] remove extra unessisary parts of gradio gui

