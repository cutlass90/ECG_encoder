import tensorflow as tf
import numpy as np

bs = 2
n = 3


mu_pl = tf.placeholder(dtype=tf.float32, shape=[None, 2])
sigma_pl = tf.placeholder(dtype=tf.float32, shape=[None, 2])
dist = tf.contrib.distributions.Normal(mu=mu_pl, sigma=sigma_pl)
a = dist.sample()
print(a)

sess = tf.InteractiveSession()

print(sess.run(a, {mu_pl:[[1.,1.],[1.,1.]], sigma_pl:[[1.,1.],[1.,1.]]}).shape)
