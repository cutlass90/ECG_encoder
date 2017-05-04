parameters = {
    'n_channels':3,
    'batch_size':64,
    'n_frames':20,
    'rr':8,
    'n_hidden_RNN':128,
    'keep_prob':1,
    'weight_decay':0.0001,
    'learn_rate_start':0.01,
    'learn_rate_end':0.0001,
    'use_delta_coding':False,
    'use_chunked_data':False,
    'verbose':True,
    'frame_weights':[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1,
                    1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
}

assert parameters['n_frames'] == len(parameters['frame_weights']), \
    'Number of frames must match with number of frame weights'

assert parameters['n_frames']%2 == 0, \
    'Number of frames must be even'