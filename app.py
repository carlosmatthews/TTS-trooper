import gradio as gr
import time
import torch
import scipy.io.wavfile
from espnet2.bin.tts_inference import Text2Speech
from espnet2.utils.types import str_or_none

tagen = 'kan-bayashi/ljspeech_vits' 
vocoder_tagen = "none" 

text2speechen = Text2Speech.from_pretrained(
    model_tag=str_or_none(tagen),
    vocoder_tag=str_or_none(vocoder_tagen),
    device="cpu",
    # Only for Tacotron 2 & Transformer
    threshold=0.5,
    # Only for Tacotron 2
    minlenratio=0.0,
    maxlenratio=10.0,
    use_att_constraint=False,
    backward_window=1,
    forward_window=3,
    # Only for FastSpeech & FastSpeech2 & VITS
    speed_control_alpha=1.0,
    # Only for VITS
    noise_scale=0.333,
    noise_scale_dur=0.333,
)


tagjp = 'kan-bayashi/jsut_full_band_vits_prosody' 
vocoder_tagjp = 'none'

text2speechjp = Text2Speech.from_pretrained(
    model_tag=str_or_none(tagjp),
    vocoder_tag=str_or_none(vocoder_tagjp),
    device="cpu",
    # Only for Tacotron 2 & Transformer
    threshold=0.5,
    # Only for Tacotron 2
    minlenratio=0.0,
    maxlenratio=10.0,
    use_att_constraint=False,
    backward_window=1,
    forward_window=3,
    # Only for FastSpeech & FastSpeech2 & VITS
    speed_control_alpha=1.0,
    # Only for VITS
    noise_scale=0.333,
    noise_scale_dur=0.333,
)

tagch = 'kan-bayashi/csmsc_full_band_vits'
vocoder_tagch = "none" 

text2speechch = Text2Speech.from_pretrained(
    model_tag=str_or_none(tagch),
    vocoder_tag=str_or_none(vocoder_tagch),
    device="cpu",
    # Only for Tacotron 2 & Transformer
    threshold=0.5,
    # Only for Tacotron 2
    minlenratio=0.0,
    maxlenratio=10.0,
    use_att_constraint=False,
    backward_window=1,
    forward_window=3,
    # Only for FastSpeech & FastSpeech2 & VITS
    speed_control_alpha=1.0,
    # Only for VITS
    noise_scale=0.333,
    noise_scale_dur=0.333,
)

def inference(text,lang):
  with torch.no_grad():
      if lang == "english":
          wav = text2speechen(text)["wav"]
          scipy.io.wavfile.write("out.wav",text2speechen.fs , wav.view(-1).cpu().numpy())
      if lang == "chinese":
          wav = text2speechch(text)["wav"]
          scipy.io.wavfile.write("out.wav",text2speechch.fs , wav.view(-1).cpu().numpy())
      if lang == "japanese":
          wav = text2speechjp(text)["wav"]
          scipy.io.wavfile.write("out.wav",text2speechjp.fs , wav.view(-1).cpu().numpy())
  return  "out.wav"
title = "ESPnet2-TTS"
description = "Gradio demo for ESPnet2-TTS: Extending the Edge of TTS Research. To use it, simply add your audio, or click one of the examples to load them. Read more at the links below."
article = "<p style='text-align: center'><a href='https://arxiv.org/abs/2110.07840' target='_blank'>ESPnet2-TTS: Extending the Edge of TTS Research</a> | <a href='https://github.com/espnet/espnet' target='_blank'>Github Repo</a></p>"

examples=[['This paper describes ESPnet2-TTS, an end-to-end text-to-speech (E2E-TTS) toolkit. ESPnet2-TTS extends our earlier version, ESPnet-TTS, by adding many new features, including: on-the-fly flexible pre-processing, joint training with neural vocoders, and state-of-the-art TTS models with extensions like full-band E2E text-to-waveform modeling, which simplify the training pipeline and further enhance TTS performance. The unified design of our recipes enables users to quickly reproduce state-of-the-art E2E-TTS results',"english"],['レシピの統一された設計により、ユーザーは最先端のE2E-TTSの結果をすばやく再現できます。また、推論用の統合Pythonインターフェースで事前にトレーニングされたモデルを多数提供し、ユーザーがベースラインサンプルを生成してデモを構築するための迅速な手段を提供します。',"japanese"],['对英语和日语语料库的实验评估表明，我们提供的模型合成了与真实情况相当的话语，达到了最先进的水平',"chinese"]]

gr.Interface(
    inference, 
    [gr.Textbox(label="input text",lines=10),gr.Radio(choices=["english", "chinese", "japanese"], type="value", value="english", label="language")], 
    gr.outputs.Audio(type="file", label="Output"),
    title=title,
    description=description,
    article=article,
    enable_queue=True,
    examples=examples
    ).launch(debug=True)
