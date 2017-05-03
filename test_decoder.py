import tensorflow as tf


batch_size = 4
n_hidden = 5
max_step = 3
embeidding_size = 6


sess = tf.InteractiveSession()

en_state = tf.ones([batch_size, n_hidden])

dec_cell = tf.contrib.rnn.GRUCell(n_hidden)
inputs = tf.ones([batch_size, max_step, embeidding_size])
seq_length = [1,2,3,2]


helper = tf.contrib.seq2seq.TrainingHelper(inputs=inputs, sequence_length=seq_length)
decoder = tf.contrib.seq2seq.BasicDecoder(
    cell=dec_cell,
    helper=helper,
    initial_state=en_state)

final_outputs, final_state = tf.contrib.seq2seq.dynamic_decode(decoder=decoder)

sess.run(tf.global_variables_initializer())
outputs_, final_state_ = sess.run([final_outputs, final_state])
# [print(var) for var in tf.trainable_variables()]

print(outputs_)
print()
print(final_state_)
