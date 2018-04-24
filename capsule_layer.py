import numpy as np
import tensorflow as tf


class CapsuleLayer(object):
    """
    Args:
        input: tensor
        num_capsules: number of capsules in the layer
        vec_length: length of output vector
        type: Fully connected or Convolution
        with_routing:
    Returns:
        4-d tensor
    """
    def __init__(self, batch_size, epsilon, num_outputs, vec_length, l_type='FC', with_routing=True):
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.vec_length = vec_length
        self.l_type = l_type
        self.with_routing = with_routing
        self.num_outputs = num_outputs

    def __call__(self, data_input, iterations, kernel_size=None, stride=None):
        if self.l_type == 'CONVOLUTION':
            self.kernel_size = kernel_size
            self.stride = stride

            if not self.with_routing:
                capsules = tf.contrib.layers.conv2d(data_input, self.num_outputs * self.vec_length,
                                                    self.kernel_size, self.stride, padding="VALID",
                                                    activation_fn=tf.nn.relu)
                capsules = tf.reshape(capsules, (self.batch_size, -1, self.vec_length, 1))
                capsules = self.squash(capsules)
                print('capsule')
                print(capsules.get_shape())
                # assert capsules.get_shape() == [self.batch_size, 1152, 8, 1]
                return capsules

        if self.l_type == 'FC':
            if self.with_routing:
                data_input = tf.reshape(data_input, shape=(self.batch_size, -1, 1, data_input.shape[-2].value, 1))
                with tf.variable_scope('routing'):
                    bias_ij = tf.constant(
                        np.zeros([self.batch_size, data_input.shape[1].value, self.num_outputs, 1, 1],
                                 dtype=np.float32))

                capsules = self.routing(data_input, iterations, bias_ij)
                capsules = tf.squeeze(capsules, axis=1)
                return capsules

    def routing(self, data_input, iterations, bias_ij):
        """
        The routing algorithm
        Returns:
            Tensor of shape [batch_size, num_capsules_l_plus_1, length(vector_vj)=16, 1]
        /Notes:
            vector_ui: the vector output of capsule i in layer l
            vector_vj: the vector output of capsule j in the layer l+1
        """
        weights = tf.get_variable('Weight', shape=(self.batch_size, 8640, 160, 2, 1),
                                  dtype=tf.float32, initializer=tf.random_normal_initializer(stddev=.01))
        biases = tf.get_variable('bias', shape=(1, 1, 2, 80, 1))

        data_input = tf.tile(data_input, [1, 1, 160, 1, 1])
        print('data_input')
        print(data_input.get_shape())
        # assert data_input.get_shape() == [self.batch_size, 1152, 10, 16, 1]
        print('weights')
        print(weights.get_shape())
        u_hat = tf.reduce_sum(weights * data_input, axis=3, keep_dims=True)
        print('u_hat')
        print(u_hat.get_shape())
        u_hat = tf.reshape(u_hat, shape=[-1, 8640, 2, 80, 1])
        print('u_hat')
        print(u_hat.get_shape())
        # assert u_hat.get_shape() == [self.batch_size, 1152, 10, 16, 1]

        u_hat_stopped = tf.stop_gradient(u_hat, name='stop_gradient')
        vector_vj = tf.placeholder(tf.float32)
        for i in range(iterations):
            with tf.variable_scope('iteration_' + str(i)):
                capsule_ij = tf.nn.softmax(bias_ij)
                print('capsule_ij')
                print(capsule_ij.get_shape())

                if i == iterations-1:
                    vector_sj = tf.multiply(capsule_ij, u_hat)
                    vector_sj = tf.reduce_sum(vector_sj, axis=1, keep_dims=True) + biases
                    print('vector_sj')
                    print(vector_sj.get_shape())
                    # assert vector_sj.get_shape() == [self.batch_size, 1, 10, 16, 1]

                    vector_vj = self.squash(vector_sj)
                    print('vector_vj')
                    print(vector_vj.get_shape())
                    # assert vector_vj.get_shape() == [self.batch_size, 1, 10, 16, 1]

                elif i < iterations-1:
                    vector_sj = tf.multiply(capsule_ij, u_hat_stopped)
                    vector_sj = tf.reduce_sum(vector_sj, axis=1, keep_dims=True) + biases
                    vector_vj = self.squash(vector_sj)

                    vector_vj_tiled = tf.tile(vector_vj, [1, 8640, 1, 1, 1])
                    product_uv = tf.reduce_sum(u_hat_stopped * vector_vj_tiled, axis=3, keep_dims=True)
                    print('product_uv')
                    print(product_uv.get_shape())
                    # assert product_uv.get_shape() == [self.batch_size, 1152, 10, 1, 1]

                    bias_ij += product_uv

        return vector_vj

    def squash(self, vector):
        """
        Squashing function
        Returns:
            A tensor with the same shape as input but "squashed" in vector length dimension
        """
        vec_squared_norm = tf.reduce_sum(tf.square(vector), -2, keep_dims=True)
        scalar_factor = vec_squared_norm / (1 + vec_squared_norm) / tf.sqrt(vec_squared_norm + self.epsilon)
        vec_squashed = scalar_factor * vector
        return vec_squashed
