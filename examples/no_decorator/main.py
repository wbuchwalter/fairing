import os

import tensorflow as tf
from six.moves import xrange  # pylint: disable=redefined-builtin
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.examples.tutorials.mnist import mnist

from fairing.train import deploy_training
from fairing.architectures.kubeflow.basic import BasicArchitecture

deploy_training(repository='<your-repository>', architecture=BasicArchitecture())


INPUT_DATA_DIR = '/tmp/tensorflow/mnist/input_data/'
MAX_STEPS = 2000
BATCH_SIZE = 100
LEARNING_RATE = 0.3
HIDDEN_1 = 128
HIDDEN_2 = 32

# HACK: Ideally we would want to have a unique subpath for each instance of the job, but since we can't
# we are instead appending HOSTNAME to the logdir
LOG_DIR = os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
                       'tensorflow/mnist/logs/fully_connected_feed/', os.getenv('HOSTNAME', ''))
MODEL_DIR = os.path.join(LOG_DIR, 'model.ckpt')


data_sets = input_data.read_data_sets(INPUT_DATA_DIR)
images_placeholder = tf.placeholder(
    tf.float32, shape=(BATCH_SIZE, mnist.IMAGE_PIXELS))
labels_placeholder = tf.placeholder(tf.int32, shape=(BATCH_SIZE))

logits = mnist.inference(images_placeholder,
                            HIDDEN_1,
                            HIDDEN_2)

loss = mnist.loss(logits, labels_placeholder)
train_op = mnist.training(loss, LEARNING_RATE)
summary = tf.summary.merge_all()
init = tf.global_variables_initializer()
saver = tf.train.Saver()
sess = tf.Session()
summary_writer = tf.summary.FileWriter(LOG_DIR, sess.graph)
sess.run(init)

data_set = data_sets.train
for step in xrange(MAX_STEPS):
    images_feed, labels_feed = data_set.next_batch(BATCH_SIZE, False)
    feed_dict = {
        images_placeholder: images_feed,
        labels_placeholder: labels_feed,
    }

    _, loss_value = sess.run([train_op, loss],
                                feed_dict=feed_dict)
    if step % 100 == 0:
        print("At step {}, loss = {}".format(step, loss_value))
        summary_str = sess.run(summary, feed_dict=feed_dict)
        summary_writer.add_summary(summary_str, step)
        summary_writer.flush()
