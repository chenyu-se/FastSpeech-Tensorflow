import numpy as np
import tensorflow as tf
import fs_hparams as fs_hp

class ScaledDotProductAttention():
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1, is_training=True):
        super().__init__()
        self.temperature = temperature
        self.padding_num = -2 ** 16 + 1
        self.attn_dropout = attn_dropout
        self.is_training = is_training
    def __call__(self, q, k, v, mask=None):

        attn = tf.matmul(q, tf.transpose(k, perm=[0, 2, 1]))
        attn = attn / self.temperature

        if mask is not None:
            print('mask dtype = ', mask.dtype)
            mask_value = tf.to_float(mask) + self.padding_num
            attn = tf.where(mask, mask_value, attn)

        attn = tf.nn.softmax(attn)
        attn = tf.layers.dropout(attn, rate=self.attn_dropout, training=self.is_training)
        output = tf.matmul(attn, v)

        return output, attn
    
    
class MultiHeadAttention():
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1, is_training=True):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.dropout = dropout
        self.is_training = is_training
        self.attention = ScaledDotProductAttention(
            temperature=np.power(d_k, 0.5), is_training=self.is_training)

    def __call__(self, q, k, v, mask=None, scope="multihead_attention"):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        d_model = self.d_model
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            sz_b, len_q = tf.shape(q)[0], tf.shape(q)[1]
            sz_b, len_k = tf.shape(k)[0], tf.shape(k)[1]
            sz_b, len_v = tf.shape(v)[0], tf.shape(v)[1]
            
            residual = q
            
            q = tf.layers.dense(q, n_head * d_k, 
                                kernel_initializer=tf.truncated_normal_initializer(mean=0,stddev=np.sqrt(2.0 / (d_model + d_k))))
            k = tf.layers.dense(k, n_head * d_k, 
                                kernel_initializer=tf.truncated_normal_initializer(mean=0,stddev=np.sqrt(2.0 / (d_model + d_k))))
            v = tf.layers.dense(v, n_head * d_v, 
                                kernel_initializer=tf.truncated_normal_initializer(mean=0,stddev=np.sqrt(2.0 / (d_model + d_v))))
            
            
            q = tf.reshape(q, [sz_b, len_q, n_head, d_k])
            k = tf.reshape(k, [sz_b, len_k, n_head, d_k])
            v = tf.reshape(v, [sz_b, len_v, n_head, d_v])
            
            
            q = tf.reshape(tf.transpose(q, perm=[2, 0, 1, 3]), [-1, len_q, d_k])
            k = tf.reshape(tf.transpose(k, perm=[2, 0, 1, 3]), [-1, len_k, d_k])
            v = tf.reshape(tf.transpose(v, perm=[2, 0, 1, 3]), [-1, len_v, d_v])
            
            
            mask = tf.tile(mask, [n_head, 1, 1])
            output, attn = self.attention(q, k, v, mask=mask)
            output = tf.reshape(output, [n_head, sz_b, len_q, d_v])
            output = tf.reshape(tf.transpose(output, perm=[1, 2, 0, 3]), [sz_b, len_q, n_head * d_v])
            
            
            output = tf.layers.dense(output, d_model)
            output = tf.layers.dropout(output, rate=self.dropout, training=self.is_training)
            #output = tf.layers.batch_normalization(output+residual, training=self.is_training)
            output = ln(output+residual)
          
        return output, attn
    
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


    
def ff(inputs, d_in, d_hid, scope="positionwise_feedforward", is_training=True):
    '''position-wise feed forward net. See 3.3
    
    inputs: A 3d tensor with shape of [N, T, C].
    num_units: A list of two integers.
    scope: Optional scope for `variable_scope`.

    Returns:
      A 3d tensor with the same shape and dtype as inputs
    '''
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # Inner layer
        outputs = tf.layers.conv1d(
			inputs,
			filters=d_hid,
			kernel_size=fs_hp.fft_conv1d_kernel,
			activation=tf.nn.relu,
			padding='same')

        # Outer layer
        outputs = tf.layers.conv1d(
			outputs,
			filters=d_in,
			kernel_size=fs_hp.fft_conv1d_kernel,
			activation=None,
			padding='same')

        outputs = tf.layers.dropout(outputs, rate=fs_hp.dropout, training=is_training,
								name='dropout_{}'.format(scope))
        # Residual connection
        outputs += inputs
        
        # Normalize
        #outputs = tf.layers.batch_normalization(outputs, training=is_training)
        outputs = ln(outputs)
    
    return outputs
