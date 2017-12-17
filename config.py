import tensorflow as tf

flags = tf.app.flags

flags.DEFINE_integer('batch_size', 256, 'batch size')
flags.DEFINE_float('m_minus', 0.01, 'm minus')
flags.DEFINE_float('m_plus', 0.99, 'm plus')
flags.DEFINE_float('lambd', 1, 'lambda')
flags.DEFINE_float('epsilon', 1e-9, 'epsilon')
flags.DEFINE_float('stddev', 1e-2, 'stddev')
flags.DEFINE_float('r_iters', 3, 'number of iterations')
flags.DEFINE_float('area', 25, 'area of activation map')
flags.DEFINE_float('lr', 1e-6, 'learning rate')

cfg = tf.app.flags.FLAGS
