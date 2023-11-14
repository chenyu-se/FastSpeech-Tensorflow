# FastSpeech-Tensorflow
FastSpeech is a state-of-art TTS(Text-to-Speech) model, and this is the implementation of FastSpeech based on Tensorflow.

For the original implementation of FastSpeech(which is based on Pytorch), see: https://github.com/xcmyz/FastSpeech

# Introduction
FastSpeech is a Text-to-Speech (TTS) model that aims to generate high-quality and natural-sounding speech. Unlike traditional TTS models, FastSpeech adopts a non-autoregressive framework, which allows for parallel generation of speech, resulting in faster inference speed.

FastSpeech consists of two main components: a duration predictor and a pitch predictor. The duration predictor predicts the length of each phoneme in the input text, while the pitch predictor predicts the fundamental frequency (F0) contour. These predictions are combined with the linguistic features to generate a mel-spectrogram, which is then converted into speech using a vocoder.

One of the key advantages of FastSpeech is its ability to generate speech in a single forward pass through the network, enabling real-time applications. Additionally, FastSpeech offers flexibility in controlling the speaking rate and pitch of the generated speech, giving users more control over the desired output.

The model is trained using a combination of a multi-speaker dataset and a parallel dataset containing text and speech pairs. Through pre-training and fine-tuning stages, FastSpeech learns to generate speech that closely matches the target speech in terms of both content and prosody.

Overall, FastSpeech demonstrates promising results in terms of both efficiency and speech quality, making it a valuable tool for various applications such as voice assistants, audiobooks, and language learning platforms.

# Model
![image](https://github.com/chenyu-se/FastSpeech-Tensorflow/assets/17283947/31f89ff4-e46a-46a4-95f1-d785360aad2b)


# Paper

[1] Shen J, Pang R, Weiss R J, et al. Natural tts synthesis by conditioning wavenet on mel spectrogram predictions[C]//2018 IEEE international conference on acoustics, speech and signal processing (ICASSP). IEEE, 2018: 4779-4783.

[2] Vaswani A, Shazeer N, Parmar N, et al. Attention is all you need[J]. Advances in neural information processing systems, 2017, 30.

[3] Ren Y, Ruan Y, Tan X, et al. Fastspeech: Fast, robust and controllable text to speech[J]. Advances in neural information processing systems, 2019, 32.

[4] Ren Y, Hu C, Tan X, et al. Fastspeech 2: Fast and high-quality end-to-end text to speech[J]. arXiv preprint arXiv:2006.04558, 2020.

Arxiv links of the papers above:

[1] Tacotron2: https://arxiv.org/abs/1712.05884

[2] Transformer: https://arxiv.org/abs/1706.03762

[3] FastSpeech: https://arxiv.org/abs/1905.09263

[4] FastSpeech2: https://arxiv.org/abs/2006.04558
