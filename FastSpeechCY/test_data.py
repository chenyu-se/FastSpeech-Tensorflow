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
from transformer import audio


def fastspeech_loss(mel, mel_postnet, duration_predictor, mel_target, duration_predictor_target):
    #mel_target.requires_grad = False
    mel_target = mel if mel_target is None else mel_target
    duration_predictor_target = duration_predictor if duration_predictor_target is None else duration_predictor_target
    
    max_len = tf.reduce_max([tf.shape(mel)[1], tf.shape(mel_postnet)[1], tf.shape(mel_target)[1]])
    mel = tf.pad(mel, [[0, 0], [0,max_len-tf.shape(mel)[1]], [0,0]], mode='CONSTANT', constant_values=0)
    mel_postnet = tf.pad(mel_postnet, [[0, 0], [0,max_len-tf.shape(mel_postnet)[1]], [0,0]], mode='CONSTANT', constant_values=0)
    mel_target = tf.pad(mel_target, [[0, 0], [0,max_len-tf.shape(mel_target)[1]], [0,0]], mode='CONSTANT', constant_values=0)
    
    mel_loss = tf.abs(mel - mel_target)
    mel_loss = tf.reduce_mean(mel_loss)
    
    mel_postnet_loss = tf.abs(mel_postnet - mel_target)
    mel_postnet_loss = tf.reduce_mean(mel_postnet_loss)

    #duration_predictor_target.requires_grad = False
    duration_predictor_target = duration_predictor_target + 1
    duration_predictor_target = tf.log(duration_predictor_target)

    #均方误差损失函数mse loss
    duration_predictor_loss = tf.reduce_mean(tf.square(duration_predictor -  duration_predictor_target))

    return mel_loss, mel_postnet_loss, duration_predictor_loss


tf.reset_default_graph()
'''logging.info("# hparams")
hparams = Hparams()
parser = hparams.parser
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
x, x_seqlens, sents1 = xs
decoder_input, y, y_seqlens, sents2 = ys

train_init_op = iter.make_initializer(train_batches)
eval_init_op = iter.make_initializer(eval_batches)






logging.info("# Load model")
FS = FastSpeech(True)
global_step = tf.train.get_or_create_global_step()
optimizer = tf.train.AdamOptimizer(learning_rate = fs_hp.learning_rate,
                                   beta1 = 0.9,
                                   beta2 = 0.98,
                                   epsilon = 1e-9)

target=tf.constant(-0.5, shape=[2, 80], dtype=tf.float32)




mel_output, mel_output_postnet, duration_predictor_output = FS(xs, ys, length_target=target)    
  
mel_loss, mel_postnet_loss, duration_predictor_loss = fastspeech_loss(
                mel_output, mel_output_postnet, duration_predictor_output, None, None)

total_loss = mel_loss + mel_postnet_loss + duration_predictor_loss       

train_op = optimizer.minimize(total_loss, global_step=global_step)

tf.summary.scalar("total_loss", total_loss)
tf.summary.scalar("global_step", global_step)
summaries = tf.summary.merge_all()'''

def prepare_run(args):
	modified_hp = hparams.parse(args.hparams)
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(args.tf_log_level)
	run_name = args.name or args.model
	log_dir = os.path.join(args.base_dir, 'logs-{}'.format(run_name))
	os.makedirs(log_dir, exist_ok=True)
	infolog.init(os.path.join(log_dir, 'Terminal_train_log'), run_name, args.slack_url)
	return log_dir, modified_hp

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default='')
    parser.add_argument('--hparams', default='',
        help='Hyperparameter overrides as a comma-separated list of name=value pairs')
    parser.add_argument('--tacotron_input', default='training_data/train.txt')
    parser.add_argument('--wavenet_input', default='tacotron_output/gta/map.txt')
    parser.add_argument('--name', help='Name of logging directory.')
    parser.add_argument('--model', default='Tacotron-2')
    parser.add_argument('--input_dir', default='training_data', help='folder to contain inputs sentences/targets')
    parser.add_argument('--output_dir', default='output', help='folder to contain synthesized mel spectrograms')
    parser.add_argument('--mode', default='synthesis', help='mode for synthesis of tacotron after training')
    parser.add_argument('--GTA', default='True', help='Ground truth aligned synthesis, defaults to True, only considered in Tacotron synthesis mode')
    parser.add_argument('--restore', type=bool, default=True, help='Set this to False to do a fresh training')
    parser.add_argument('--summary_interval', type=int, default=250,
		help='Steps between running summary ops')
    parser.add_argument('--checkpoint_interval', type=int, default=5000,
		help='Steps between writing checkpoints')
    parser.add_argument('--eval_interval', type=int, default=10000,
		help='Steps between eval on test data')
    parser.add_argument('--tacotron_train_steps', type=int, default=100, help='total number of tacotron training steps') #默认值原本为100000 
    parser.add_argument('--wavenet_train_steps', type=int, default=750000, help='total number of wavenet training steps')
    parser.add_argument('--tf_log_level', type=int, default=1, help='Tensorflow C++ log level.')
    parser.add_argument('--slack_url', default=None, help='slack webhook notification destination link')
    args = parser.parse_args()
    log_dir, hparams = prepare_run(args)
    
    
    input_path = os.path.join(args.base_dir, args.tacotron_input)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    
    
    
    #准备数据，建立模型，优化器和损失函数
    tf.set_random_seed(hparams.tacotron_random_seed)
    coord = tf.train.Coordinator()
    with tf.variable_scope('datafeeder') as scope:
        feeder = Feeder(coord, input_path, hparams)
        
        
        
    inputs, mel_targets, alignment_targets = feeder.inputs, feeder.mel_targets, feeder.alignment_targets
    
    
    
    
    
    
    
    
    
    step = 0
    eval_dir = os.path.join(log_dir, 'eval-dir')
    eval_wav_dir = os.path.join(eval_dir, 'wavs')
    
    target_max_length = tf.shape(mel_targets)[1]
    predicted_max_length = tf.reduce_sum(alignment_targets + 1, axis=1)
    predicted_max_length = tf.reduce_max(predicted_max_length)
    
    
    
    
    #训练
    with tf.Session(config=config) as sess:
        try:
            sess.run(tf.global_variables_initializer())
            feeder.start_threads(sess)
            while not coord.should_stop() and step < args.tacotron_train_steps:
                tgt, prd, mel = sess.run([inputs, alignment_targets, mel_targets[0]])
                print('the target is : ', tgt)
                print('the algnment is : ', prd)
                print('the mel_target is : ', mel)
        except Exception as e:
            #log('Exiting due to exception: {}'.format(e), slack=True)
            traceback.print_exc()
            coord.request_stop(e)
    print('training finished')
    
    
if __name__ == '__main__':
    main()
    
    
    