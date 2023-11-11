import fs_hparams as fs_hp
from transformer.hparams import Hparams
from transformerCY.Models import Transformer
from transformer.modules import Postnet
from Networks import LengthRegulator
import tensorflow as tf


class FastSpeech:
    def __init__(self, is_training=True):
        self.trans_hp = Hparams().parser.parse_args()
        self.trans = Transformer()
        self.encoder = self.trans.encode
        self.length_regulator = LengthRegulator(is_training)
        self.decoder = self.trans.decode
        #self.mel_linear = Linear(hp.decoder_output_size, hp.num_mels)
        self.postnet = Postnet(is_training)
        self.is_training = is_training
        self.num_mels = fs_hp.num_mels
        
    def __call__(self, xs, ys, mel_max_length=None, length_target=None, alpha=1.0):
        with tf.variable_scope("fast_speech", reuse=tf.AUTO_REUSE):
            encoder_output, sents1, encoder_mask = self.encoder(xs, training=self.is_training)
            if self.is_training:
                length_regulator_output, decoder_pos, duration_predictor_output = self.length_regulator(
                    encoder_output,
                    encoder_mask,
                    length_target,
                    alpha,
                    mel_max_length)
                
                print('length_regulator_output.shape = ',length_regulator_output.get_shape().as_list())
                decoder_output = self.decoder(ys, length_regulator_output, decoder_pos, training=self.is_training)
                print(length_regulator_output)
                print(decoder_pos)
                print(duration_predictor_output)
                mel_output = tf.layers.dense(decoder_output,units=self.num_mels,use_bias=True)
                mel_output_postnet = self.postnet(mel_output) + mel_output
                
                return mel_output, mel_output_postnet, duration_predictor_output
            else:
                length_regulator_output, decoder_pos, duration_predictor_output = self.length_regulator(
                    encoder_output, encoder_mask, alpha=alpha)
                
                decoder_output = self.decoder(ys, length_regulator_output, decoder_pos, training=self.is_training)
                
                mel_output = tf.layers.dense(decoder_output,units=self.num_mels,use_bias=True)
                mel_output_postnet = self.postnet(mel_output) + mel_output
                
                
                return mel_output, mel_output_postnet, duration_predictor_output