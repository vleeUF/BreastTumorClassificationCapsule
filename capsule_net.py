import tensorflow as tf
import numpy as np
from data_prep import split_training, batch, create_data_sets
from tqdm import tqdm
from capsule_layer import CapsuleLayer


class CapsuleNet(object):
    def __init__(self, batch_size, learning_rate, regularization_scale, epsilon, is_training=True):
        self.graph = tf.Graph()
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.regularization_scale = regularization_scale
        self.epsilon = epsilon
        self.arg_max_idx = tf.placeholder(tf.int32)
        self.decoded = tf.placeholder(tf.float32)
        self.v_length = tf.placeholder(tf.float32)
        self.masked_v = tf.placeholder(tf.float32)
        self.caps2 = tf.placeholder(tf.float32)
        self.accuracy = tf.placeholder(tf.float32)
        self.train_summary = tf.placeholder(tf.float32)

        with self.graph.as_default():
            if is_training:

                self.x, self.labels = batch(self.batch_size)
                self.x = tf.cast(self.x, tf.float32)
                self.labels = tf.cast(self.labels, tf.int32)
                self.y = tf.one_hot(self.labels, depth=2, axis=1)

                self.build()
                total_loss, m_loss, r_loss = self.loss(self.regularization_scale)
                self.summary(total_loss, m_loss, r_loss)

                # Train
                global_step = tf.Variable(0, name='global_step', trainable=False)
                self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
                self.train_op = self.optimizer.minimize(total_loss, global_step=global_step)

            else:
                self.x = tf.placeholder(tf.float32, shape=(None, 70, 46, 1))
                self.y = tf.placeholder(tf.int32, shape=(self.batch_size, ))
                # self.y = tf.reshape(self.labels, shape=(self.batch_size, 4, 1))
                self.build()

    def build(self):
        with tf.variable_scope("c1_layer"):
            c1 = tf.contrib.layers.conv2d(self.x, 100, 9, stride=1, padding='Valid')
            print('c1')
            print(c1.get_shape())
            # assert c1.get_shape() == [self.batch_size, 20, 20, 256]

        with tf.variable_scope('PrimaryCaps_layer'):
            primary_caps = CapsuleLayer(self.batch_size, self.epsilon, 16, 2, l_type='CONVOLUTION', with_routing=False)
            caps1 = primary_caps(c1, 3, kernel_size=4, stride=2)
            print('caps1: ')
            print(caps1.get_shape())
            # assert caps1.get_shape() == [self.batch_size, 1152, 8, 1]

        with tf.variable_scope('FC_Caps_layer'):
            fc_caps = CapsuleLayer(self.batch_size, self.epsilon, 2, 16, l_type='FC', with_routing=True)
            self.caps2 = fc_caps(caps1, 3)
            print('caps2')
            print(self.caps2.get_shape())
            # assert self.caps2.get_shape() == [self.batch_size, 10, 16, 1]

        with tf.variable_scope('Masking'):
            self.v_length = tf.sqrt(tf.reduce_sum(tf.square(self.caps2), axis=2, keep_dims=True) + self.epsilon)
            sm_v = tf.nn.softmax(self.v_length)
            print('sm_v')
            print(sm_v.get_shape())
            # assert sm_v.get_shape() == [self.batch_size, 4, 1, 1]

            self.arg_max_idx = tf.to_int32(tf.argmax(sm_v, axis=1))
            print('arg_max_idx')
            print(self.arg_max_idx.get_shape())
            # assert self.arg_max_idx.get_shape() == [self.batch_size, 1, 1]
            self.arg_max_idx = tf.reshape(self.arg_max_idx, shape=(self.batch_size, ))

            self.masked_v = tf.multiply(tf.squeeze(self.caps2), tf.reshape(self.y, (-1, 2, 1)))
            self.v_length = tf.sqrt(tf.reduce_sum(tf.square(self.caps2), axis=2, keep_dims=True) + self.epsilon)

        with tf.variable_scope('Decoder'):
            v_j = tf.reshape(self.masked_v, shape=(self.batch_size, -1))
            fc1 = tf.contrib.layers.fully_connected(v_j, num_outputs=512)
            assert fc1.get_shape() == [self.batch_size, 512]
            fc2 = tf.contrib.layers.fully_connected(fc1, num_outputs=1024)
            assert fc2.get_shape() == [self.batch_size, 1024]
            self.decoded = tf.contrib.layers.fully_connected(fc2, num_outputs=9660, activation_fn=tf.nn.relu)

    def loss(self, regularization_scale):

        with tf.variable_scope('margin_loss'):
            m_loss = self.margin_loss()

        with tf.variable_scope('reconstruction_loss'):
            r_loss = self.reconstruction_loss(origin=self.x, decoded=self.decoded)

        total_loss = m_loss + regularization_scale * r_loss

        return total_loss, m_loss, r_loss

    def summary(self, total_loss, m_loss, r_loss):
        tf.summary.scalar('margin_loss', m_loss)
        tf.summary.scalar('reconstruction_loss', r_loss)
        tf.summary.scalar('total_loss', total_loss)
        self.train_summary = tf.summary.merge_all()

        correct_prediction = tf.equal(tf.to_int32(self.labels), self.arg_max_idx)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def margin_loss(self, m_plus=0.9, m_minus=0.1, lambda_val=0.5):
        max_left = tf.square(tf.maximum(0., m_plus - self.v_length))
        max_right = tf.square(tf.maximum(0., self.v_length - m_minus))
        # assert max_left.get_shape() == (self.batch_size, 4)
        # assert max_right.get_shape() == (self.batch_size, 4)

        t_k = self.y
        l_k = t_k * max_left + lambda_val * (1-t_k) * max_right
        m_loss = tf.reduce_mean(tf.reduce_sum(l_k, axis=1))
        return m_loss

    def reconstruction_loss(self, origin, decoded):
        origin = tf.reshape(origin, shape=(self.batch_size, -1))
        decoded = tf.reshape(decoded, shape=(self.batch_size, -1))
        r_loss = tf.reduce_mean(tf.square(decoded - origin))
        return r_loss

    def train(self, supervisor, epoch):

        train_x, train_y, num_tr_batch, val_x, val_y, num_val_batch = split_training(.8, .125, self.batch_size)
        train_sum_freq = 100
        val_sum_freq = 100
        save_freq = 3

        config = tf.ConfigProto()
        with supervisor.managed_session(config=config) as sess:

            for e in range(epoch):
                print("Training for epoch %d/%d:" % (e, epoch))
                if supervisor.should_stop():
                    print('supervisor stopped!')
                    break
                print(num_tr_batch)
                for step in tqdm(range(num_tr_batch-120), total=num_tr_batch, leave=False, unit='b'):

                    print(step)
                    global_step = epoch * num_tr_batch + step

                    if global_step % train_sum_freq == 0:
                        _, loss, train_accuracy, summary = sess.run([self.train_op, self.loss,
                                                                     self.accuracy, self.train_summary])
                        assert not np.isnan(loss), "Error! loss is nan"
                        supervisor.summary_writer.add_summary(summary, global_step)

                    else:
                        sess.run(self.train_op)

                    if val_sum_freq != 0 and global_step % val_sum_freq == 0:
                        v_accuracy = 0
                        for v in range(num_val_batch):
                            start = v * self.batch_size
                            end = start + self.batch_size
                            accuracy = sess.run(self.accuracy, {self.x: val_x[start:end], self.y: val_y[start:end]})
                            v_accuracy += accuracy
                        v_accuracy = v_accuracy / num_val_batch
                        print(v_accuracy)
                if (epoch + 1) % save_freq == 0:
                    supervisor.saver.save(sess, "log_directory" + "/model_epoch_%04d" % epoch)

    def evaluation(self, supervisor):
        test_x, test_y, num_test_batches = create_data_sets(.8, self.batch_size)
        with supervisor.managed_session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:

            supervisor.saver.restore(sess, tf.train.latest_checkpoint('log_directory'))
            tf.logging.info('Model ready...')

            test_accuracy = 0
            for i in tqdm(range(num_test_batches), total=num_test_batches, leave=False, unit='b'):
                start = i * self.batch_size
                end = start + self.batch_size
                print(start)
                print(end)
                print(test_x.shape)
                accuracy = sess.run(self.accuracy, {self.x: test_x[start:end], self.labels: test_y[start:end]})
                test_accuracy += accuracy
                print(test_accuracy)
            test_accuracy = test_accuracy / num_test_batches
            print(test_accuracy)
            print('Test Accuracy: %6.2f' % test_accuracy)









