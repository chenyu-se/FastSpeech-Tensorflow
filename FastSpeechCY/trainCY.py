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
from FastSpeech2 import FastSpeech
from tqdm import tqdm
from transformer.feeder import Feeder
from transformer import audio
import numpy as np
from utils import plot
from datetime import datetime


os.environ['CUDA_VISIBLE_DEVICES']='0'
infolog.init(r'logs-Tacotron-2/logs/fs_log1.txt', None)
log = infolog.log

def fastspeech_loss(mel, mel_postnet, duration_predictor, mel_target, duration_predictor_target):
    #mel_target.requires_grad = False
    mel_target = mel if mel_target is None else mel_target
    duration_predictor_target = duration_predictor if duration_predictor_target is None else duration_predictor_target
    
    '''
    max_len = tf.reduce_max([tf.shape(mel)[1], tf.shape(mel_postnet)[1], tf.shape(mel_target)[1]])
    mel = tf.pad(mel, [[0, 0], [0,max_len-tf.shape(mel)[1]], [0,0]], mode='CONSTANT', constant_values=0)
    mel_postnet = tf.pad(mel_postnet, [[0, 0], [0,max_len-tf.shape(mel_postnet)[1]], [0,0]], mode='CONSTANT', constant_values=0)
    mel_target = tf.pad(mel_target, [[0, 0], [0,max_len-tf.shape(mel_target)[1]], [0,0]], mode='CONSTANT', constant_values=0)
    '''
    
    
    min_len = tf.reduce_min([tf.shape(mel)[1], tf.shape(mel_postnet)[1], tf.shape(mel_target)[1]])
    mel, mel_postnet, mel_target = mel[:, 0:min_len, :], mel_postnet[:, 0:min_len, :], mel_target[:, 0:min_len, :]
    
    
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



def add_optimizer(global_step, hp, loss):
    if hp.tacotron_decay_learning_rate:
        decay_steps = hp.tacotron_decay_steps
        decay_rate = hp.tacotron_decay_rate
        learning_rate = _learning_rate_decay(hp, hp.tacotron_initial_learning_rate, global_step, decay_steps, decay_rate)
    else:
        learning_rate = tf.convert_to_tensor(hp, hp.tacotron_initial_learning_rate)
    
    #FastSpeech所使用的learningrate
    n_warmup_steps = fs_hp.n_warm_up_step
    n_current_steps = tf.to_float(global_step + 1)
    init_lr = np.power(fs_hp.word_vec_dim, -0.5)
    scale1 = 1 / tf.sqrt(n_current_steps)
    scale2 = np.power(n_warmup_steps, -1.5) * n_current_steps
    learning_rate = init_lr * tf.reduce_min([scale1, scale2])
    
    
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate,
                                   beta1 = 0.9,
                                   beta2 = 0.98,
                                   epsilon = 1e-9)
    
    
    '''gradients, variables = zip(*optimizer.compute_gradients(loss))
    gradients = gradients
    #Just for causion
    #https://github.com/Rayhane-mamah/Tacotron-2/issues/11
    if hp.tacotron_clip_gradients:
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, 1.)
    else:
        clipped_gradients = gradients
    
    
    
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        train_op = optimizer.apply_gradients(zip(clipped_gradients, variables),
					global_step=global_step)'''
    train_op = optimizer.minimize(loss, global_step=global_step)
    
    return train_op, optimizer



def _learning_rate_decay(hp, init_lr, global_step, decay_steps, decay_rate):
    lr = tf.train.exponential_decay(init_lr,
			global_step - hp.tacotron_start_decay, #lr = 1e-3 at step 50k
			decay_steps,
			decay_rate, #lr = 1e-5 around step 310k
			name='lr_exponential_decay')
    
    return tf.minimum(tf.maximum(lr, hp.tacotron_final_learning_rate), init_lr)
        




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






def time_string():
	return datetime.now().strftime('%Y-%m-%d %H:%M')



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
    parser.add_argument('--tacotron_train_steps', type=int, default=500000, help='total number of tacotron training steps') #默认值原本为100000 
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
    x, x_seqlens, sents1 = (inputs, None, None)
    decoder_input, y, y_seqlens, sents2 = (mel_targets, None, None, None)
    xs = (x, x_seqlens, sents1)
    ys = (decoder_input, y, y_seqlens, sents2)
    
    
    eval_inputs, eval_mel_targets, eval_alignment_targets = feeder.eval_inputs, feeder.eval_mel_targets, feeder.eval_alignment_targets
    eval_x, eval_x_seqlens, eval_sents1 = (eval_inputs, None, None)
    eval_decoder_input, eval_y, eval_y_seqlens, eval_sents2 = (eval_mel_targets, None, None, None)
    eval_xs = (eval_x, eval_x_seqlens, eval_sents1)
    eval_ys = (eval_decoder_input, eval_y, eval_y_seqlens, eval_sents2)
    
    
    
    
    
    
    m = FastSpeech(True)
    eval_m = FastSpeech(False)
    global_step = tf.train.get_or_create_global_step()
    optimizer = tf.train.AdamOptimizer(learning_rate = fs_hp.learning_rate,
                                       beta1 = 0.9,
                                       beta2 = 0.98,
                                       epsilon = 1e-9)
    
    optimizer2 = tf.train.AdadeltaOptimizer(learning_rate = fs_hp.learning_rate)
    
    mel_output, mel_output_postnet, duration_predictor_output = m(xs, ys,
                                                                  mel_max_length = tf.shape(mel_targets)[1], 
                                                                  length_target=alignment_targets, 
                                                                  alpha=1.0)
    
    eval_mel_output, eval_mel_output_postnet, eval_duration_predictor_output = eval_m(eval_xs, eval_ys,
                                                                  mel_max_length = None, 
                                                                  length_target = None, 
                                                                  alpha=1.0)
    
    
    
    
    mel_loss, mel_postnet_loss, duration_predictor_loss = fastspeech_loss(
                    mel_output, 
                    mel_output_postnet, 
                    duration_predictor_output, 
                    mel_targets, 
                    alignment_targets)
    
    
    
    eval_mel_loss, eval_mel_postnet_loss, eval_duration_predictor_loss = fastspeech_loss(
                    eval_mel_output, 
                    eval_mel_output_postnet, 
                    eval_duration_predictor_output, 
                    eval_mel_targets, 
                    eval_alignment_targets)
    
    total_loss = mel_loss + mel_postnet_loss + duration_predictor_loss
    
    #train_op = optimizer.minimize(total_loss, global_step=global_step)

    
    train_op, optimizer = add_optimizer(global_step, hparams, total_loss)
    
    
    
    
    step = 0
    eval_dir = os.path.join(log_dir, 'eval-dir')
    eval_plot_dir = os.path.join(eval_dir, 'plots')
    eval_wav_dir = os.path.join(eval_dir, 'wavs')
    os.makedirs(eval_dir, exist_ok=True)
    os.makedirs(eval_plot_dir, exist_ok=True)
    os.makedirs(eval_wav_dir, exist_ok=True)
    
    
    
    saver = tf.train.Saver(max_to_keep=5)
    save_dir = os.path.join(log_dir, 'fastspeech_pretrained')
    checkpoint_path = os.path.join(save_dir, 'fastspeech_model.ckpt')
    
    
    
    #训练
    with tf.Session(config=config) as sess:
        try:
            sess.run(tf.global_variables_initializer())
            #saved model restoring
            if args.restore:
                # Restore saved model if the user requested it, default = True
                try:
                    checkpoint_state = tf.train.get_checkpoint_state(save_dir)
                    
                    if (checkpoint_state and checkpoint_state.model_checkpoint_path):
                        log('Loading checkpoint {}'.format(checkpoint_state.model_checkpoint_path), slack=True)
                        saver.restore(sess, checkpoint_state.model_checkpoint_path)
                    else:
                        log('No model to load at {}'.format(save_dir), slack=True)

                except tf.errors.OutOfRangeError as e:
                    log('Cannot restore checkpoint: {}'.format(e), slack=True)
            else:
                log('Starting new training!', slack=True)
            
            feeder.start_threads(sess)
            while not coord.should_stop() and step < args.tacotron_train_steps:
                step, l1, l2, l3, opt = sess.run([global_step, mel_loss, mel_postnet_loss, duration_predictor_loss, train_op])
                #eval_l1, eval_l2, eval_l3 = sess.run([eval_mel_loss, eval_mel_postnet_loss, eval_duration_predictor_loss])
                #print('train: ', step, l1, l2, l3, l1+l2+l3)
                #print('eval: : ', step, eval_l1, eval_l2, eval_l3, eval_l1+eval_l2+eval_l3)
                if step % 10 == 0:
                    _lr = sess.run(optimizer._lr)
                    e_l1, e_l2, e_l3 = sess.run([eval_mel_loss, eval_mel_postnet_loss, eval_duration_predictor_loss])
                    print('step: {}, train_loss: {}, {}, {}, {}'.format(step, l1, l2, l3, l1+l2+l3))
                    print('step: {}, eval_loss: {}, {}, {}, {}'.format(step, e_l1, e_l2, e_l3, e_l1+e_l2+e_l3))
                    print('learning_rate: {}'.format(_lr))
                if step % 2000 == 1999:
                    #Save model and current global step
                    saver.save(sess, checkpoint_path, global_step=global_step)
                    
                    
                    text_t, mel_p, mel_t, alg, alg_t = sess.run([inputs[0], mel_output_postnet[0], mel_targets[0], duration_predictor_output[0], alignment_targets[0]])
                    log('text: {}'.format(text_t))
                    log('out_put: {}'.format(mel_p))
                    log('target: {}'.format(mel_t))
                    log('duration_predictor_output: {}'.format(np.expm1(alg)))
                    log('alignment_target: {}'.format(alg_t))
                    wav = audio.inv_mel_spectrogram(mel_p.T)
                    audio.save_wav(wav, os.path.join(eval_wav_dir, 'step-{}-eval-wave-from-mel.wav'.format(step)))
                    
                    plot.plot_alignment(alg, os.path.join(eval_plot_dir, 'step-{}-train-align.png'.format(step)),
						title='{}, {}, step={}, loss={:.5f}'.format(args.model, time_string(), step, duration_predictor_loss),
						max_len=None)
                    plot.plot_alignment(alg_t, os.path.join(eval_plot_dir, 'step-{}-train-align_target.png'.format(step)),
						title='{}, {}, step={}, loss={:.5f}'.format(args.model, time_string(), step, duration_predictor_loss),
						max_len=None)
                    #print(step + 'steps have run.')
        except Exception as e:
            #log('Exiting due to exception: {}'.format(e), slack=True)
            traceback.print_exc()
            coord.request_stop(e)
    print('training finished')
    
    
if __name__ == '__main__':
    main()
    
    
    
