import tensorflow as tf
import fs_hparams as hp
import numpy as np


class LengthRegulator:
    """ Length Regulator """

    def __init__(self, is_training):
        super(LengthRegulator, self).__init__()

        self.duration_predictor = DurationPredictor(is_training)
        
        self.training=is_training

    def LR(self, encoder_output, duration_predictor_output, alpha, mel_max_length=None):
        
        
        predicted_lens = tf.to_int32(tf.round(duration_predictor_output*alpha+1))
        expand_lens = tf.reduce_sum(predicted_lens, axis=1)
        max_len = tf.reduce_max(expand_lens)
        if mel_max_length is not None:
            #max_len = tf.cond(max_len>mel_max_length, lambda: max_len, lambda: mel_max_length)
            max_len = mel_max_length
        pad_lens = -expand_lens + max_len
        
        output, dec_pos = tf.map_fn(self.expand, 
                        (encoder_output, predicted_lens, pad_lens), 
                        dtype=(encoder_output.dtype, tf.bool))
        
        
        return output, dec_pos
    
   
    
    def expand(self, elems):
        one_batch, predicted, pad_len = elems
        
        out = tf.TensorArray(size=0, dtype=one_batch.dtype, dynamic_size=True)
        i = tf.constant(0)
        j = tf.constant(0)
        n = tf.shape(predicted)[0]
        sum_len = predicted[0]
        #i, sum_len = self.find_next_predicted(i, predicted)
        

        
        
        out, i, j, n, sum_len, one_batch, predicted = tf.while_loop(self.cond1, 
                                                                    self.body1, 
                                                                    [out, i, j, n, sum_len, one_batch, predicted])        

        out = out.stack()
        
        pos = tf.concat([tf.range(tf.shape(out)[0], dtype=tf.int32) + 1,
                         tf.zeros(pad_len, dtype=tf.int32)], 0)
        pos = tf.math.equal(pos, 0)
        out = tf.pad(out, [[0,pad_len],[0,0]], mode='CONSTANT', constant_values=0.0)
        
        return (out, pos)


    '''def find_next_predicted(self, i, predicted):
        n = tf.shape(predicted)[0]
        res, n = tf.while_loop(lambda i, n: tf.cond(i<n,
                                                  lambda: predicted[i]==0,
                                                  lambda: False),
                             lambda i, n: (i+1, n), [i, n])
        return (res, tf.cond(res>=n, lambda: predicted[res-1], lambda: predicted[res]))'''
        
        
        

    def cond1(self, out, i, j, n, sum_len, one_batch, predicted):
        #self.find_next_predicted(i, predicted)[0]
        return i < n
    
    def body1(self, out, i, j, n, sum_len, one_batch, predicted):
        #i = self.find_next_predicted(i, predicted)[0]
        index = i
          
        out = out.write(j, one_batch[index])
        j += 1
        
        i, sum_len = tf.cond(j >= sum_len, 
            lambda: (index+1, tf.cond(index+1<n,
                                      lambda: sum_len+predicted[index+1], 
                                      lambda: sum_len)),
            lambda: (index, sum_len))
        
        return out, i, j, n, sum_len, one_batch, predicted       

    def __call__(self, encoder_output, encoder_output_mask, target=None, alpha=1.0, mel_max_length=None):
        duration_predictor_output = self.duration_predictor(
            encoder_output, encoder_output_mask)
        # print(duration_predictor_output)

        if self.training:
            target = duration_predictor_output if target is None else target
            output, decoder_pos = self.LR(
                encoder_output, target, alpha, mel_max_length)

            return output, decoder_pos, duration_predictor_output
        else:
            duration_predictor_output = tf.exp(duration_predictor_output)
            duration_predictor_output = duration_predictor_output - 1
            # print(duration_predictor_output)

            output, decoder_pos = self.LR(
                encoder_output, duration_predictor_output, alpha)

            return output, decoder_pos









































class DurationPredictor:
    """ Duration Predictor """

    def __init__(self, is_training, activation=tf.nn.relu, scope=None):

        self.scope = 'DurationPredictor' if scope is None else scope
        self.input_size = hp.encoder_output_size
        self.filter_size = hp.duration_predictor_filter_size
        self.kernel = hp.duration_predictor_kernel_size
        self.conv_output_size = hp.duration_predictor_filter_size
        self.dropout = hp.dropout
        self.is_training = is_training
        self.activation = activation
        self.relu = tf.nn.relu
        self.training=is_training

    #is_training=True时，输出张量的shape为(batch_size,encoder_output_size)
    def __call__(self, encoder_output, encoder_output_mask):
        with tf.variable_scope(self.scope):
            encoder_output = encoder_output * encoder_output_mask
    
            out = conv1d(encoder_output,
                         self.kernel,
                         self.filter_size,
                         self.activation,
                         self.is_training,
                         self.dropout,
                         'conv_layer_{}_'.format(1)+self.scope,
                         padding='same')
            out = conv1d(out,
                         self.kernel,
                         self.filter_size,
                         self.activation,
                         self.is_training,
                         self.dropout,
                         'conv_layer_{}_'.format(2)+self.scope,
                         padding='same')
            
            
            out = linear(out, 1, 'linear_layer_{}_'.format(1)+self.scope)
            
            out = out * encoder_output_mask[:, :, 0:1]
            
            #out = out * encoder_output_mask
            
            out = self.relu(out)
            
            print(out.shape)
            
            out = tf.squeeze(out)

    
            if not self.training:
                out = tf.expand_dims(out,0)
            
            return out



def ln(inputs, epsilon = 1e-8, scope="ln"):
    '''Applies layer normalization. See https://arxiv.org/abs/1607.06450.
    inputs: A tensor with 2 or more dimensions, where the first dimension has `batch_size`.
    epsilon: A floating number. A very small number for preventing ZeroDivision Error.
    scope: Optional scope for `variable_scope`.
      
    Returns:
      A tensor with the same shape and data dtype as `inputs`.
    '''
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]
    
        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta= tf.get_variable("beta", params_shape, initializer=tf.zeros_initializer())
        gamma = tf.get_variable("gamma", params_shape, initializer=tf.ones_initializer())
        normalized = (inputs - mean) / ( (variance + epsilon) ** (.5) )
        outputs = gamma * normalized + beta
        
    return outputs


def conv1d(inputs, kernel_size, channels, activation, is_training, drop_rate, scope, padding='same'):
	with tf.variable_scope(scope):
		conv1d_output = tf.layers.conv1d(
			inputs,
			filters=channels,
			kernel_size=kernel_size,
			activation=None,
			padding=padding)
		#batched = tf.layers.batch_normalization(conv1d_output, training=is_training)
		batched = ln(conv1d_output)
		activated = activation(batched)
		return tf.layers.dropout(activated, rate=drop_rate, training=is_training,
								name='dropout_{}'.format(scope))
        
        
def linear(inputs, units, scope, activation=None,bias=True):
    with tf.variable_scope(scope):
        linear_output = tf.layers.dense(inputs,units=units,use_bias=bias)
        if activation!=None:
            linear_output = activation(linear_output)
        return linear_output






'''class Conv:
    """
    Convolution Module
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=1,
                 stride=1,
                 padding=0,
                 dilation=1,
                 bias=True,
                 w_init='relu',
                 name=None):
        """
        :param in_channels: dimension of input
        :param out_channels: dimension of output
        :param kernel_size: size of kernel
        :param stride: size of stride
        :param padding: size of padding
        :param dilation: dilation rate
        :param bias: boolean. if True, bias is included.
        :param w_init: str. weight inits with xavier initialization.
        """

        self.scope = 'Conv' if name is None else name
        self.conv=tf.layers.conv1d(filters=out_channels,
                                   kernel_size=kernel_size,
                                   strides=stride,
                                   padding=padding,
                                   dilation_rate=dilation,
                                   use_bias=bias,
                                   name='C')

    def __call__(self, x):
       with tf.variable_scope(self.scope):
            return self.conv(x)

class Linear:
    """
    Linear Module
    """

    def __init__(self, in_dim, out_dim, bias=True, w_init='relu',name=None):
        """
        :param in_dim: dimension of input
        :param out_dim: dimension of output
        :param bias: boolean. if True, bias is included.
        :param w_init: str. weight inits with xavier initialization.
        """
        self.scope = 'Linear' if name is None else name
        self.units=out_dim
        """
        建立一个全连接层,不指定kernel_initializer时则会使用默认的Xavier初始化方法
        """
        self.Linear_layer = tf.layers.Dense(units=self.units, name='L')
    def __call__(self, x):
        with tf.variable_scope(self.scope):
            return self.linear_layer(x)
'''
