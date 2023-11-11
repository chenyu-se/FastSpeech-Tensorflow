# -*- coding: utf-8 -*-
# /usr/bin/python3
'''
Feb. 2019 by kyubyong park.
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer

Transformer network
'''
import tensorflow as tf

#from transformer.data_load import load_vocab
import numpy as np
from tqdm import tqdm
from text.symbols import symbols
import fs_hparams as fs_hp
import logging
from transformerCY.SubLayers import MultiHeadAttention, ff
import transformerCY.Constants as Constants

logging.basicConfig(level=logging.INFO)

def get_non_pad_mask(seq):
    #assert seq.dim() == 2
    non_pad_mask = tf.to_float(tf.not_equal(seq, Constants.PAD))
    return tf.expand_dims(non_pad_mask, axis=-1)


def get_attn_key_pad_mask(seq_k, seq_q):
    ''' For masking out the padding part of key sequence. '''

    # Expand to fit the shape of key query attention matrix.
    len_q = tf.shape(seq_q)[1]
    padding_mask = tf.equal(seq_k, Constants.PAD)
    padding_mask = tf.expand_dims(padding_mask, axis=1)
    padding_mask = tf.tile(padding_mask, [1, len_q, 1])
    return padding_mask





def get_token_embeddings(vocab_size, num_units, zero_pad=True):
    '''Constructs token embedding matrix.
    Note that the column of index 0's are set to zeros.
    vocab_size: scalar. V.
    num_units: embedding dimensionalty. E.
    zero_pad: Boolean. If True, all the values of the first row (id = 0) should be constant zero
    To apply query/key masks easily, zero pad is turned on.

    Returns
    weight variable: (V, E)
    '''
    with tf.variable_scope("shared_weight_matrix", reuse=tf.AUTO_REUSE  ):
        embeddings = tf.get_variable('weight_mat',
                                   dtype=tf.float32,
                                   shape=(vocab_size, num_units),
                                   initializer=tf.contrib.layers.xavier_initializer())
        if zero_pad:
            embeddings = tf.concat((tf.zeros(shape=[1, num_units]),
                                    embeddings[1:, :]), 0)
    return embeddings


def positional_encoding(inputs,
                        maxlen,
                        masking=True,
                        scope="positional_encoding"):
    '''Sinusoidal Positional_Encoding. See 3.5
    inputs: 3d tensor. (N, T, E)
    maxlen: scalar. Must be >= T
    masking: Boolean. If True, padding positions are set to zeros.
    scope: Optional scope for `variable_scope`.

    returns
    3d tensor that has the same shape as inputs.
    '''

    print('inputs_shape:', inputs.get_shape().as_list())
    E = inputs.get_shape().as_list()[-1] # static
    N, T = tf.shape(inputs)[0], tf.shape(inputs)[1] # dynamic
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # position indices
        position_ind = tf.tile(tf.expand_dims(tf.range(T), 0), [N, 1]) # (N, T)

        # First part of the PE function: sin and cos argument
        position_enc = np.array([
            [pos / np.power(10000, (i-i%2)/E) for i in range(E)]
            for pos in range(maxlen)])

        # Second part, apply the cosine to even columns and sin to odds.
        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1
        position_enc = tf.convert_to_tensor(position_enc, tf.float32) # (maxlen, E)

        # lookup
        outputs = tf.nn.embedding_lookup(position_enc, position_ind)

        # masks
        if masking:
            outputs = tf.where(tf.equal(inputs, 0), inputs, outputs)

        return tf.to_float(outputs)





class Transformer:
    '''
    xs: tuple of
        x: int32 tensor. (N, T1)
        x_seqlens: int32 tensor. (N,)
        sents1: str tensor. (N,)
    ys: tuple of
        decoder_input: int32 tensor. (N, T2)
        y: int32 tensor. (N, T2)
        y_seqlen: int32 tensor. (N, )
        sents2: str tensor. (N,)
    training: boolean.
    '''
    def __init__(self):
        #self.token2idx, self.idx2token = load_vocab(hp.vocab)
        self.embeddings = get_token_embeddings(len(symbols)+1, fs_hp.word_vec_dim, zero_pad=True)
        self.enc_att = MultiHeadAttention(fs_hp.encoder_head, fs_hp.word_vec_dim, 64, 64, dropout=fs_hp.dropout)
        self.dec_att = MultiHeadAttention(fs_hp.decoder_head, fs_hp.word_vec_dim, 64, 64, dropout=fs_hp.dropout)
    def encode(self, xs, training=True):
        '''
        Returns
        memory: encoder outputs. (N, T1, d_model)
        '''
        with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
            x, seqlens, sents1 = xs

            # embedding
            enc = tf.nn.embedding_lookup(self.embeddings, x) # (N, T1, d_model)
            n_position = fs_hp.max_sep_len + 1

            slf_attn_mask = get_attn_key_pad_mask(seq_k=x, seq_q=x)
            non_pad_mask = get_non_pad_mask(x)
            
            enc += positional_encoding(enc, n_position)

            ## Blocks
            for i in range(fs_hp.encoder_n_layer):
                with tf.variable_scope("num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
                    # self-attention
                    enc, enc_slf_attn = self.enc_att(
                                enc, enc, enc, mask=slf_attn_mask)
                    enc *= non_pad_mask
                    # feed forward
                    enc = ff(enc, fs_hp.word_vec_dim, fs_hp.encoder_conv1d_filter_size, is_training=training)
                    enc *= non_pad_mask
        memory = enc
        return memory, sents1, non_pad_mask

    def decode(self, ys, memory, src_masks, training=True):
        '''
        memory: encoder outputs. (N, T1, d_model)
        src_masks: (N, T1)

        Returns
        logits: (N, T2, V). float32.
        y_hat: (N, T2). int32
        y: (N, T2). int32
        sents2: (N,). string.
        '''
        with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
            decoder_inputs, y, seqlens, sents2 = ys
            
            
            enc_pos = src_masks
            dec = memory
            n_position = fs_hp.max_sep_len + 1
            # embedding
            slf_attn_mask = get_attn_key_pad_mask(seq_k=enc_pos, seq_q=enc_pos)
            non_pad_mask = get_non_pad_mask(enc_pos)

            dec += positional_encoding(dec, n_position)

            # Blocks
            for i in range(fs_hp.decoder_n_layer):
                with tf.variable_scope("num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
                    dec, enc_slf_attn = self.dec_att(
                                dec, dec, dec, mask=slf_attn_mask)
                    dec *= non_pad_mask
                    ### Feed Forward
                    dec = ff(dec, fs_hp.word_vec_dim, fs_hp.decoder_conv1d_filter_size, is_training=training)
                    dec *= non_pad_mask

        return dec

