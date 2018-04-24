import tensorflow as tf
from capsule_net import CapsuleNet


flags = tf.app.flags

###################
# Hyperparameters #
###################

# Margin Loss
flags.DEFINE_float('m_plus', 0.9, 'the parameter of m plus')
flags.DEFINE_float('m_minus', 0.1, 'the parameter of m minus')
flags.DEFINE_float('lambda_val', 0.5, 'weight of the loss')

# Training
flags.DEFINE_integer('batch_size', 16, 'batch_size')
flags.DEFINE_float('learning_rate', 1e-3, 'learning rate of optimizer')
flags.DEFINE_float('epsilon', 1e-9, 'epsilon')
flags.DEFINE_integer('epoch', 20, 'epoch')
flags.DEFINE_float('regularization_scale', 0.092, 'regularization coefficient for reconstruction loss')

# Settings
flags.DEFINE_boolean('is_training', False, 'train or predict phase')

FLAGS = flags.FLAGS

"""
def save(sess, checkpoint_dir):
    saver = tf.train.Saver()
    print("Saving...")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    saver.save(sess, checkpoint_dir)
"""


def main(_):

    # Build Graph
    tf.logging.info('Loading graph...')
    model = CapsuleNet(FLAGS.batch_size, FLAGS.learning_rate,
                       FLAGS.regularization_scale, FLAGS.epsilon)
    tf.logging.info('Graph loaded')

    # Start Session
    sv = tf.train.Supervisor(graph=model.graph,
                             logdir='\log_directory',
                             save_model_secs=0)

    if FLAGS.is_training:
        tf.logging.info('Start training...')
        model.train(sv, FLAGS.epoch)
        tf.logging.info('Training done')
    else:
        model.evaluation(sv)


if __name__ == "__main__":
    tf.app.run()
