import tensorflow as tf
import os
import argparse
import logging
import traceback
from transformer import infolog
import fs_hparams as fs_hp
from hparams import hparams
from transformer.hparams import Hparams
from transformer.utils import save_hparams, save_variable_specs, get_hypotheses, calc_bleu
from transformer.data_load import get_batch
from FastSpeech import FastSpeech
from tqdm import tqdm
from transformer.feeder import Feeder

def fastspeech_loss(mel, mel_postnet, duration_predictor, mel_target, duration_predictor_target):
    #mel_target.requires_grad = False
    mel_target = tf.zeros(tf.shape(mel)) if mel_target is None else mel_target
    duration_predictor_target = tf.zeros(tf.shape(duration_predictor)) if duration_predictor_target is None else duration_predictor_target
    
    min_len = min(mel.shape[1], mel_postnet.shape[1], mel_target.shape[1])
    mel = mel[:, 0:min_len, :]
    mel_postnet = mel_postnet[:, 0:min_len, :]
    mel_target = mel_target[:, 0:min_len, :]
    
    mel_loss = tf.abs(mel - mel_target)
    mel_loss = tf.reduce_mean(mel_loss)
    
    mel_postnet_loss = tf.abs(mel_postnet - mel_target)
    mel_postnet_loss = tf.reduce_mean(mel_postnet_loss)

    #duration_predictor_target.requires_grad = False
    duration_predictor_target = duration_predictor_target + 1
    duration_predictor_target = tf.log(duration_predictor_target)

    #均方误差损失函数mse loss
    duration_predictor_loss = tf.reduce_sum(tf.square(duration_predictor -  duration_predictor_target))

    return mel_loss, mel_postnet_loss, duration_predictor_loss

tf.reset_default_graph()


print(tf.__version__)

'''one_batch = tf.stack([tf.zeros(2,dtype=tf.int32), tf.ones(2,dtype=tf.int32)])
pad_length = tf.constant([2,3])
#lens = tf.stack([tf.unstack(one_batch), tf.unstack(pad_length)])

#print(tf.Session().run(lens))
out = tf.map_fn(lambda x: tf.tile(x[0], [x[1]]), (one_batch, pad_length), dtype=tf.int32, infer_shape=False)

print(tf.Session().run(out))'''
logging.info("# hparams")

parser = Hparams().parser
hp = parser.parse_args()
save_hparams(hp, hp.logdir)

logging.info("# Prepare train/eval batches")
train_batches, num_train_batches, num_train_samples = get_batch(hp.train1, hp.train2,
                                             hp.maxlen1, hp.maxlen2,
                                             hp.vocab, hp.batch_size,
                                             shuffle=True)
eval_batches, num_eval_batches, num_eval_samples = get_batch(hp.eval1, hp.eval2,
                                             100000, 100000,
                                             hp.vocab, hp.batch_size,
                                             shuffle=False)

# create a iterator of the correct shape and type
iter = tf.data.Iterator.from_structure(train_batches.output_types, train_batches.output_shapes)
xs, ys = iter.get_next()

train_init_op = iter.make_initializer(train_batches)
eval_init_op = iter.make_initializer(eval_batches)

#xs = x, x_seqlens, sents1
#ys = decoder_input, y, y_seqlens, sents2
m = FastSpeech(True)
global_step = tf.train.get_or_create_global_step()
optimizer = tf.train.AdamOptimizer(learning_rate = fs_hp.learning_rate,
                                   beta1 = 0.9,
                                   beta2 = 0.98,
                                   epsilon = 1e-9)
mel_output, mel_output_postnet, duration_predictor_output = m(xs, ys, length_target=None)

mel_loss, mel_postnet_loss, duration_predictor_loss = fastspeech_loss(
                mel_output, mel_output_postnet, duration_predictor_output, None, None)

total_loss = mel_loss + mel_postnet_loss + duration_predictor_loss       

train_op = optimizer.minimize(total_loss, global_step=global_step)

'''tf.summary.scalar("total_loss", total_loss)
tf.summary.scalar("global_step", global_step)
train_summaries = tf.summary.merge_all()'''

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(train_init_op)
    total_steps = hp.num_epochs * num_train_batches
    
    _gs = sess.run(global_step)
    
    
    
    for i in tqdm(range(_gs, total_steps+1)):
        _, _gs, _loss = sess.run([train_op, global_step, total_loss])
        print(_gs, _loss)
        if _gs and _gs % num_train_batches == 0:
            sess.run(train_init_op)

tf.reset_default_graph()