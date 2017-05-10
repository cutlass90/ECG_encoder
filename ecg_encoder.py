import os
import time
import math
import itertools as it

import tensorflow as tf
import numpy as np
from tqdm import tqdm

import ecg_encoder_tools as utils




class ECGEncoder(object):

    def __init__(self, n_frames, n_channel, n_hidden_RNN, reduction_ratio,
        frame_weights, do_train, n_parts=None):

        self.n_frames = n_frames
        self.n_channel = n_channel
        self.n_hidden_RNN = n_hidden_RNN
        self.reduction_ratio = reduction_ratio
        self.frame_weights = frame_weights
        self.n_parts = n_parts
        self.do_train = do_train
        if n_parts == None:
            self.create_graph()
        else:
            self.create_inference_graph()
        if do_train: self.create_optimizer_graph(self.cost)
        os.makedirs('summary', exist_ok=True)
        sub_d = len(os.listdir('summary'))
        self.train_writer = tf.summary.FileWriter(logdir = 'summary/'+str(sub_d))
        self.merged = tf.summary.merge_all()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config = config)
        self.sess.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver(var_list=tf.global_variables(),
                                    max_to_keep = 1000)
        
    # --------------------------------------------------------------------------
    def __enter__(self):
        return self


    # --------------------------------------------------------------------------
    def __exit__(self, exc_type, exc_val, exc_tb):
        tf.reset_default_graph()
        if self.sess is not None:
            self.sess.close()
        

    # --------------------------------------------------------------------------
    def create_graph(self):
        print('Creat graph')

        self.inputs,\
        self.sequence_length,\
        self.keep_prob,\
        self.weight_decay,\
        self.learn_rate = self.input_graph() # inputs shape is #b*n_f x h1 x c1

        # Encoder
        convo = self.convo_graph(self.inputs) #b*n_f x h2 x c2
        # print('convo', convo)

        seq_l = tf.cast((self.sequence_length/self.reduction_ratio), tf.int32)
        frame_embs = self.compress_frames(convo, seq_l, n_layers=2) # b*n_f x hRNN
        # print('frame_embs', frame_embs)# b x n_f x hRNN

        frame_embs = tf.reshape(frame_embs, [-1, self.n_frames, self.n_hidden_RNN])
        Z_l, Z_r = self.encode_to_Z(inputs=frame_embs, n_hidden=2*self.n_hidden_RNN) #b x 2*hRNN
        # print('Z left', Z_l)
        # print('Z right', Z_r)

        
        mu_l, sigma_l = tf.split(Z_l, 2, axis=1)
        mu_r, sigma_r = tf.split(Z_r, 2, axis=1)
        sigma_l = tf.nn.softplus(sigma_l)
        sigma_r = tf.nn.softplus(sigma_r)
        Z_l = self.sample_Z(mu=mu_l, sigma=sigma_l) #b x hRNN
        Z_r = self.sample_Z(mu=mu_r, sigma=sigma_r) #b x hRNN
            

        self.Z = tf.concat((Z_l,Z_r), axis=1)
        tf.summary.histogram('Z hist', self.Z)


        
        # Decoder
        r_frame_embs = self.decode_from_Z(encoded_states=[Z_l,Z_r]) # b x n_f x hRNN
        # print('r_frame_embs', r_frame_embs)

        r_frame_embs = tf.reshape(r_frame_embs, [-1, self.n_hidden_RNN])# b*n_f x hRNN
        r_convo = self.decompress_frames(encoded_state=r_frame_embs,
            seq_lengths=seq_l, n_layers=1) #b*n_f x h2 x c2
        # print('r_convo', r_convo)

        self.r_inputs = self.deconvo_graph(r_convo) #b*n_f x h1 x c1
        # print('r_inputs', self.r_inputs)



        self.cost = self.create_cost_graph(original=self.inputs,
            recovered=self.r_inputs, frame_weights=self.frame_weights, mu_l=mu_l,
            sigma_l=sigma_l, mu_r=mu_r, sigma_r=sigma_r)
        
        
        print('Done!')

    # --------------------------------------------------------------------------
    def create_inference_graph(self):
        print('Creat inference graph')

        self.inputs,\
        self.sequence_length,\
        self.keep_prob,\
        self.weight_decay,\
        self.learn_rate = self.input_graph() # inputs shape is # n_f*n_p x h1 x c1
        self.inputs = tf.reshape(self.inputs, [self.n_frames*self.n_parts, -1, self.n_channel])

        # Encoder
        convo = self.convo_graph(self.inputs) # n_f*n_p x h2 x c2
        # print('convo', convo)

        seq_l = tf.cast((self.sequence_length/self.reduction_ratio), tf.int32)
        frame_embs = self.compress_frames(convo, seq_l, n_layers=2) # n_f*n_p x hRNN

        frame_embs = tf.reshape(frame_embs,[1, self.n_frames*self.n_parts, self.n_hidden_RNN])
        # print('frame_embs', frame_embs)# 1 x n_p*n_f x hRNN
        n_Z = (self.n_parts-1)*self.n_frames + 1
        b_frame_embs = tf.concat([frame_embs[:,i:i+self.n_frames,:] for i in range(n_Z)], 0) # n_Z x n_f x hRNN
        # print('b_frame_embs',b_frame_embs)
        Z_l, Z_r = self.encode_to_Z(b_frame_embs, n_hidden=2*self.n_hidden_RNN) #n_Z x 2*hRNN

        mu_l, sigma_l = tf.split(Z_l, 2, axis=1)
        mu_r, sigma_r = tf.split(Z_r, 2, axis=1)
        sigma_l = tf.nn.softplus(sigma_l)
        sigma_r = tf.nn.softplus(sigma_r)
        Z_l = self.sample_Z(mu=mu_l, sigma=sigma_l) #b x hRNN
        Z_r = self.sample_Z(mu=mu_r, sigma=sigma_r) #b x hRNN

        self.Z = tf.concat([Z_l, Z_r], axis=1) # n_Z x 2*hRNN
        # print('Z ', self.Z)
        
        print('Done!')
    # --------------------------------------------------------------------------
    def input_graph(self):
        print('\tinput_graph')
        inputs = tf.placeholder(tf.float32,
            shape=[None, None, self.n_channel],
            name='inputs') #b*n_f x h x c (h is variable value)

        sequence_length = tf.placeholder(tf.int32, shape=[None],
            name='sequence_length') # b*n_f

        keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        weight_decay = tf.placeholder(tf.float32, name='weight_decay')

        learn_rate = tf.placeholder(tf.float32, name='learn_rate')

        # self.batch_size = tf.size(sequence_length)

        return inputs, sequence_length, keep_prob, weight_decay, learn_rate


    # --------------------------------------------------------------------------
    def convo_graph(self, inputs):
        print('\tconvo_graph')
        with tf.variable_scope('convo_graph'):
            convo1 = self.conv_1d(inputs,
                kernelShape=[2, 3, 16],
                strides=2,
                activation=tf.nn.elu,
                keep_prob=self.keep_prob)
            convo2 = self.conv_1d(convo1,
                kernelShape=[2, 16, 32],
                strides=2,
                activation=tf.nn.elu,
                keep_prob=self.keep_prob)
            convo3 = self.conv_1d(convo2,
                kernelShape=[2, 32, 64],
                strides=2,
                activation=tf.nn.elu,
                keep_prob=self.keep_prob)
        return convo3


    # --------------------------------------------------------------------------
    def conv_1d(self, inputs, kernelShape, strides=1, activation = None, keep_prob = None):
        print('\t\tconv_1d')
        with tf.variable_scope(None, 'conv_1d') as sc:
            # inputs to convo need [batch, in_width, in_channels]
            # kernels shape must be [filter_width, in_channels, out_channels]
            kernel = tf.get_variable('kernel', shape=kernelShape,
                initializer=tf.contrib.layers.xavier_initializer())
            bias = tf.get_variable('bias', shape=[kernelShape[2]],
                initializer=tf.constant_initializer(0.0))
            convo = tf.nn.bias_add(tf.nn.conv1d(inputs, filters=kernel,
                stride=strides, padding="SAME"), bias)
            if activation is not None:
                convo = activation(convo, name='activation')
            if keep_prob is not None:
                convo = tf.nn.dropout(convo, keep_prob=keep_prob, name="keep_prob")
            return convo


    # --------------------------------------------------------------------------
    def compress_frames(self, inputs, sequence_length, n_layers):
        print('\tcompress_frames')
        with tf.variable_scope('compress_frames'):
            # inputs b*n_f x h x c (h is variable value)
            # sequence_length b*n_f
            cell = tf.contrib.rnn.GRUCell(self.n_hidden_RNN)

            fw_cell = tf.contrib.rnn.MultiRNNCell([cell]*n_layers)
            fw_cell = tf.contrib.rnn.DropoutWrapper(fw_cell,
                output_keep_prob=self.keep_prob)

            bw_cell = tf.contrib.rnn.MultiRNNCell([cell]*n_layers)
            bw_cell = tf.contrib.rnn.DropoutWrapper(bw_cell,
                output_keep_prob=self.keep_prob)

            outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_cell,
                cell_bw=bw_cell,
                inputs=inputs,
                sequence_length=sequence_length,
                dtype=tf.float32)
        return states[0][-1] + states[1][-1] #shape b*n_f x hRNN


    # --------------------------------------------------------------------------
    def encode_to_Z(self, inputs, n_hidden):
        # inputs b x n_f x hRNN
        print('\tencode_to_Z')

        Z_l = self.RNN(inputs[:,:self.n_frames//2,:], scope='encode_Z_left',
            n_layers=2, n_hidden=n_hidden) # b x n_hidden

        reversed_data = tf.reverse(inputs[:,self.n_frames//2:,:], axis=[1])
        Z_r = self.RNN(reversed_data, scope='encode_Z_right',
            n_layers=2, n_hidden=n_hidden) # b x n_hidden
        
        return Z_l, Z_r


    # --------------------------------------------------------------------------
    def sample_Z(self, mu, sigma):
        dist = tf.contrib.distributions.Normal(mu, sigma)
        return dist.sample()
        



    # --------------------------------------------------------------------------
    def RNN(self, inputs, scope, n_layers, n_hidden):
        print('\t\t'+scope)
        with tf.variable_scope(scope):
            cell = tf.contrib.rnn.GRUCell(n_hidden)

            cell = tf.contrib.rnn.MultiRNNCell([cell]*n_layers)
            cell = tf.contrib.rnn.DropoutWrapper(cell,
                output_keep_prob=self.keep_prob)

            outputs, states = tf.nn.dynamic_rnn(
                cell=cell,
                inputs=inputs,
                dtype=tf.float32)
        return states[-1] #shape b x hRNN


    # --------------------------------------------------------------------------
    def decode_from_Z(self, encoded_states):
        print('\tdecode_from_Z')
        Z_l, Z_r = encoded_states
        r_frames_l = self.dRNN(Z_l, n_hidden=self.n_hidden_RNN, scope='decode_Z_l') # b x n_f//2 x hRNN
        r_frames_r = self.dRNN(Z_r, n_hidden=self.n_hidden_RNN, scope='decode_Z_r') # b x n_f//2 x hRNN
        r_frames_r = tf.reverse(r_frames_r, axis=[1]) # b x n_f//2 x hRNN
        r_frames = tf.concat([r_frames_l, r_frames_r], axis=1) # b x n_f x hRNN
        return r_frames       


    # --------------------------------------------------------------------------
    def dRNN(self, encoded_state, n_hidden, scope):
        print('\t\t'+scope)
        with tf.variable_scope(scope):
            decoder_fn = utils.simple_decoder_fn_train_(encoded_state)
            dec_cell = tf.contrib.rnn.GRUCell(n_hidden)

            sl = tf.tile([self.n_frames//2], [tf.shape(encoded_state)[0]])
            recover, _, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(
                cell=dec_cell,
                decoder_fn=decoder_fn,
                inputs=None,
                sequence_length=sl)

            return recover



    # --------------------------------------------------------------------------
    def decompress_frames(self, encoded_state, seq_lengths, n_layers):
        print('\tdecompress_frames')
        # first step is zero-vectors

        with tf.variable_scope('decompress_frames'):
            decoder_fn = utils.simple_decoder_fn_train_(encoded_state)
            dec_cell = tf.contrib.rnn.GRUCell(self.n_hidden_RNN)

            recover, _, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(
                cell=dec_cell,
                decoder_fn=decoder_fn,
                inputs=None,
                sequence_length=seq_lengths)

            return recover


    # --------------------------------------------------------------------------
    def deconvo_graph(self, inputs):
        # inputs [b, h, c]
        print('\tdeconvo_graph')
        deconvo1 = self.deconv_1d(inputs=inputs, filters=32, kernel_size=2, strides=2)
        deconvo2 = self.deconv_1d(inputs=deconvo1, filters=16, kernel_size=2, strides=2)
        deconvo3 = self.deconv_1d(inputs=deconvo2, filters=3, kernel_size=2, strides=2)
        return deconvo3


    # --------------------------------------------------------------------------
    def deconv_1d(self, inputs, filters, kernel_size, strides, activation = None,
        keep_prob = None):
        """
        Args:
            inputs: tensor of shape [batch, width, in_channels]
            filters: integer, the dimensionality of the output space
            (i.e. the number output of filters in the convolution).
            kernel_size: integer specifying the spatial dimensions of of the filters
            strides: integer specifying the strides of the convolution.
        """
        print('\t\tdeconv_1d')
        inputs = tf.expand_dims(inputs, 2)
        with tf.variable_scope(None, 'deconv_1d') as sc:
            convo = tf.layers.conv2d_transpose(inputs=inputs,
                filters=filters,
                kernel_size=[kernel_size,1],
                strides=[strides, 1],
                padding='same',
                activation=activation,
                kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))
            convo = tf.squeeze(convo, axis=2)
            if keep_prob is not None:
                convo = tf.nn.dropout(convo, keep_prob=keep_prob, name="keep_prob")
            return convo


    # --------------------------------------------------------------------------
    def create_cost_graph(self, original, recovered, frame_weights, mu_l,
            sigma_l, mu_r, sigma_r):
        print('\tcreate_cost_graph')

        a = tf.reduce_mean(tf.square(original - recovered), [1,2]) # b*n_f
        b = tf.reduce_mean(tf.reshape(a, [-1, self.n_frames]), 0) #n_f
        self.mse = tf.reduce_mean(b*self.frame_weights)

        self.L2_loss = self.weight_decay*sum([tf.reduce_mean(tf.square(var))
            for var in tf.trainable_variables()])

        self.KL_loss = tf.reduce_mean(self.KL_loss(mu_l, sigma_l) +
            self.KL_loss(mu_r, sigma_r))

        tf.summary.scalar('MSE', self.mse)
        tf.summary.scalar('L2 loss', self.L2_loss)
        tf.summary.scalar('KL loss', self.KL_loss)
        return self.mse + self.L2_loss + self.KL_loss

    # --------------------------------------------------------------------------
    def KL_loss(self, mu, sigma):
        loss = tf.log(1/sigma) + (sigma*sigma+mu*mu)/2 - 0.5
        return loss


    # --------------------------------------------------------------------------
    def create_optimizer_graph(self, cost):
        print('create_optimizer_graph')
        with tf.variable_scope('optimizer_graph'):
            optimizer = tf.train.AdamOptimizer(self.learn_rate)
            self.train = optimizer.minimize(cost)

        
    #---------------------------------------------------------------------------  
    def save_model(self, path = 'beat_detector_model', step = None):
        p = self.saver.save(self.sess, path, global_step = step)
        print("\tModel saved in file: %s" % p)

    #---------------------------------------------------------------------------
    def load_model(self, path):
        #path is path to file or path to directory
        #if path it is path to directory will be load latest model
        load_path = os.path.splitext(path)[0]\
        if os.path.isfile(path) else tf.train.latest_checkpoint(path)
        print('try to load {}'.format(load_path))
        self.saver.restore(self.sess, load_path)
        print("Model restored from file %s" % load_path)


    #---------------------------------------------------------------------------
    def train_(self, data_loader,  keep_prob, weight_decay, learn_rate_start,
        learn_rate_end, n_iter, save_model_every_n_iter, path_to_model):
        print('\n\n\n\t----==== Training ====----')
        #try to load model
        try:
            self.load_model(os.path.dirname(path_to_model))
        except:
            print('Can not load model {0}, starting new train'.format(path_to_model))
            
        start_time = time.time()
        b = math.log(learn_rate_start/learn_rate_end, n_iter) 
        a = learn_rate_start*math.pow(1, b)
        for current_iter in tqdm(range(n_iter)):
            learn_rate = a/math.pow((current_iter+1), b)
            batch = data_loader.get_batch()
            feedDict = {self.inputs : batch['normal_data'],
                        self.sequence_length : batch['sequence_length'],
                        self.keep_prob : keep_prob,
                        self.weight_decay : weight_decay,
                        self.learn_rate : learn_rate}
            _, summary = self.sess.run([self.train, self.merged], feed_dict=feedDict)
            self.train_writer.add_summary(summary, current_iter)

            if (current_iter+1) % save_model_every_n_iter == 0:
                self.save_model(path = path_to_model, step = current_iter+1)

        self.save_model(path = path_to_model, step = current_iter+1)
        print('\nTrain finished!')
        print("Training time --- %s seconds ---" % (time.time() - start_time))


    # --------------------------------------------------------------------------
    def predict(self, path_to_file, path_to_save, path_to_model, use_delta_coding):

        print('\n\n\n\t----==== Predicting ====----')
        self.load_model(path_to_model)
        
        data = np.load(path_to_file).item()

        gen = utils.step_generator(data,
                   n_frames = 1,
                   overlap = self.n_frames-1,
                   get_data = not use_delta_coding,
                   get_delta_coded_data = use_delta_coding,
                   rr = self.reduction_ratio,
                   get_events = False)
        
        list_of_res = []

        forward_pass_time = 0
        for current_iter in tqdm(it.count()):
            try:
                batch = next(gen)
            except StopIteration:
                break
            feedDict = {self.inputs : batch['normal_data'], #1*n_f x h x c (h is variable value)
                        self.sequence_length : batch['sequence_length'],
                        self.keep_prob : 1}
            start_time = time.time()
            res = self.sess.run(self.r_inputs, feed_dict=feedDict) #n_f x h x c
            forward_pass_time = forward_pass_time + (time.time() - start_time)

            result = np.empty([0,self.n_channel])
            original = np.empty([0,self.n_channel])
            for f in range(self.n_frames):
                h = batch['seq_l']
                r = np.reshape(res[f,:h[f], :], [-1, self.n_channel])
                o = np.reshape(batch['normal_data'][f,:h[f], :], [-1, self.n_channel])
                result = np.concatenate((result, r), 0)
                original = np.concatenate((original, o), 0)
            list_of_res.append({'original':original, 'recovered':result})

        if path_to_save is not None:
            np.save(path_to_save, list_of_res)
            print('\nfile saved ', path_to_save)


        return list_of_res

    # --------------------------------------------------------------------------
    def get_Z(self, data, path_to_save, path_to_model, use_delta_coding):
        """ Return Z-code for all beat in data.

        Args:
            data: may be either path to *.npy file or dict with data
        """
        self.load_model(path_to_model)

        data = np.load(data).item() if isinstance(data, str) else data

        gen = utils.step_generator(data,
                   n_frames = (self.n_parts-1)*self.n_frames+1,
                   overlap = self.n_frames-1,
                   get_data = not use_delta_coding,
                   get_delta_coded_data = use_delta_coding,
                   rr = self.reduction_ratio,
                   get_events = False)
        
        result = np.empty([0, 2*self.n_hidden_RNN])

        forward_pass_time = 0
        for current_iter in tqdm(it.count()):
            try:
                batch = next(gen)
            except StopIteration:
                break
            feedDict = {self.inputs : batch['normal_data'], #n_p*n_f x h x c (h is variable value)
                        self.sequence_length : batch['sequence_length'],
                        self.keep_prob : 1}
            start_time = time.time()
            res = self.sess.run(self.Z, feed_dict=feedDict) # (n_p-1)*n_f+1 x 2*hRNN
            forward_pass_time = forward_pass_time + (time.time() - start_time)
            result = np.concatenate((result, res), 0)
        print('result shape', result.shape)

        # zero padding
        n_beats = len(data['beats'])
        end_pad = n_beats - self.n_frames//2 - result.shape[0]
        result = np.concatenate(
            (np.zeros([self.n_frames//2, 2*self.n_hidden_RNN]),
            result,
            np.zeros([end_pad, 2*self.n_hidden_RNN])), axis=0)

        if path_to_save is not None:
            np.save(path_to_save, result)
            print('\nfile saved ', path_to_save)

        return result


# testing #####################################################################################################################
if __name__ == '__main__':
    """
    ECGEncoder(
        n_frames=20,
        n_channel=3,
        n_hidden_RNN=128,
        reduction_ratio=8,
        frame_weights=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1,
                    1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
        do_train=False)
    """
    ECGEncoder(n_frames=20, n_channel=3, n_hidden_RNN=128, reduction_ratio=8,
        frame_weights=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1,
                    1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
        n_parts=None, do_train=True)

