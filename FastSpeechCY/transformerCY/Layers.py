import tensorflow as tf
from transformerCY.SubLayers import multihead_attention, ff




class Prenet:
	"""Two fully connected layers used as an information bottleneck for the attention.
	"""
	def __init__(self, is_training, layers_sizes=[256, 256], drop_rate=0.5, activation=tf.nn.relu, scope=None):
		"""
		Args:
			layers_sizes: list of integers, the length of the list represents the number of pre-net
				layers and the list values represent the layers number of units
			activation: callable, activation functions of the prenet layers.
			scope: Prenet scope.
		"""
		super(Prenet, self).__init__()
		self.drop_rate = drop_rate

		self.layers_sizes = layers_sizes
		self.activation = activation
		self.is_training = is_training

		self.scope = 'prenet' if scope is None else scope

	def __call__(self, inputs):
		x = inputs

		with tf.variable_scope(self.scope):
			for i, size in enumerate(self.layers_sizes):
				dense = tf.layers.dense(x, units=size, activation=self.activation,
					name='dense_{}'.format(i + 1))
				#The paper discussed introducing diversity in generation at inference time
				#by using a dropout of 0.5 only in prenet layers (in both training and inference).
				x = tf.layers.dropout(dense, rate=self.drop_rate, training=True,
					name='dropout_{}'.format(i + 1) + self.scope)
		return x


class Postnet:
	"""Postnet that takes final decoder output and fine tunes it (using vision on past and future frames)
	"""
	def __init__(self, is_training, 
                 n_mel_channels=80,
                 postnet_embedding_dim=512,
                 postnet_kernel_size=5,
                 postnet_n_convolutions=5, 
                 activation=tf.nn.tanh, scope=None):
		"""
		Args:
			is_training: Boolean, determines if the model is training or in inference to control dropout
			kernel_size: tuple or integer, The size of convolution kernels
			channels: integer, number of convolutional kernels
			activation: callable, postnet activation function for each convolutional layer
			scope: Postnet scope.
		"""
		super(Postnet, self).__init__()
		self.is_training = is_training

		self.kernel_size = 5
		self.channels = postnet_embedding_dim
		self.activation = activation
		self.scope = 'postnet_convolutions' if scope is None else scope
		self.postnet_num_layers = 5
		self.drop_rate = 0.5

	def __call__(self, inputs):
		with tf.variable_scope(self.scope):
			x = inputs
			for i in range(self.postnet_num_layers - 1):
				x = conv1d(x, self.kernel_size, self.channels, self.activation,
					self.is_training, self.drop_rate, 'conv_layer_{}_'.format(i + 1)+self.scope)
			x = conv1d(x, self.kernel_size, self.channels, lambda _: _, self.is_training, self.drop_rate,
				'conv_layer_{}_'.format(5)+self.scope)
		return x




def conv1d(inputs, kernel_size, channels, activation, is_training, drop_rate, scope, padding=0):
	with tf.variable_scope(scope):
		conv1d_output = tf.layers.conv1d(
			inputs,
			filters=channels,
			kernel_size=kernel_size,
			activation=None,
			padding=padding)
		batched = tf.layers.batch_normalization(conv1d_output, training=is_training)
		activated = activation(batched)
		return tf.layers.dropout(activated, rate=drop_rate, training=is_training,
								name='dropout_{}'.format(scope))
        
        
def linear(inputs, units, scope, activation=None,bias=True):
    with tf.variable_scope(scope):
        linear_output = tf.layers.dense(inputs,units=units,use_bias=bias)
        if activation!=None:
            linear_output = activation(linear_output)
        return linear_output
    

def fft_block(enc_input, num_units, non_pad_mask=None, slf_attn_mask=None):
    enc_output, enc_slf_attn = multihead_attention(enc_input, enc_input, enc_input, slf_attn_mask)
    enc_output *= non_pad_mask
    enc_output = ff(enc_output, num_units)
    enc_output *= non_pad_mask
    return enc_output, enc_slf_attn