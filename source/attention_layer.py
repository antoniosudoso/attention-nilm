import tensorflow as tf


class AttentionLayer(tf.keras.Model):

    def __init__(self, units):
        super(AttentionLayer, self).__init__()
        weight_initializer = 'he_normal'
        self.W = tf.keras.layers.Dense(units, kernel_initializer=weight_initializer)
        self.V = tf.keras.layers.Dense(1, kernel_initializer=weight_initializer)

    def call(self, encoder_output, **kwargs):
        # encoder_output shape == (batch_size, seq_length, latent_dim)
        # score shape == (batch_size, seq_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, seq_length, units)
        score = self.V(tf.nn.tanh(self.W(encoder_output)))

        # attention_weights shape == (batch_size, seq_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * encoder_output
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights
