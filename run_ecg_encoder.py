import os

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from ecg_encoder_parameters import parameters as PARAM
import ecg_encoder_tools as utils
from ecg_encoder import ECGEncoder
import ecg



print('\n\n\n\t----==== Import parameters ====----')
with open('ecg_encoder_parameters.py', 'r') as param:
    print(param.read())

os.makedirs('summary/', exist_ok = True)



# path_to_train_data = '../data/little/'
# path_to_train_data      = '../../data/train/chunked/'
# path_to_train_data      = '../../../ECG_DATA/all/chunked_data/'
path_to_train_data = '../data/small_set/'
# path_eval_cost_data     = '../../../ECG_DATA/ECG_DATA_1000samples_2/test/'
# path_to_predict_data    = path_to_train_data
path_to_predictions     = 'predictions/'
os.makedirs(path_to_predictions, exist_ok = True)
n_iter_train            = 4000
# n_iter_eval             = 10000
save_model_every_n_iter = 10000
path_to_model = 'models/ecg_encoder'


gen_params = dict(n_frames = PARAM['n_frames'],
                overlap = 0,
                get_data = not(PARAM['use_delta_coding']),
                get_delta_coded_data = PARAM['use_delta_coding'],
                get_events = False,
                rr = PARAM['rr'])

# Initialize data loader for training
data_loader = utils.LoadDataFileShuffling(batch_size=PARAM['batch_size'],
                                    path_to_data=path_to_train_data,
                                    gen=utils.step_generator,
                                    gen_params=gen_params,
                                    verbose=PARAM['verbose'])

""""
# Train model
with ECGEncoder(
    n_frames=PARAM['n_frames'],
    n_channel=PARAM['n_channels'],
    n_hidden_RNN=PARAM['n_hidden_RNN'],
    reduction_ratio=PARAM['rr'],
    use_true_inps=True,
    do_train=True) as ecg_encoder:
    
    
    ecg_encoder.train_(
        data_loader = data_loader,
        keep_prob=PARAM['keep_prob'],
        weight_decay=PARAM['weight_decay'],
        learn_rate_start=PARAM['learn_rate_start'],
        learn_rate_end=PARAM['learn_rate_end'],
        n_iter=n_iter_train,
        save_model_every_n_iter=save_model_every_n_iter,
        path_to_model=path_to_model)

    # [print(var) for var in tf.trainable_variables()]


# Predictions
path='../data/little/AAO1CMED2K865.npy'
f_name = ecg.utils.get_file_name(path)
with ECGEncoder(
    n_frames=PARAM['n_frames'],
    n_channel=PARAM['n_channels'],
    n_hidden_RNN=PARAM['n_hidden_RNN'],
    reduction_ratio=PARAM['rr'],
    use_true_inps=True,
    do_train=False) as ecg_encoder:

    ecg_encoder.predict(
        path_to_file=path,
        path_to_save=path_to_predictions+f_name+'_pred.npy',
        path_to_model=os.path.dirname(path_to_model),
        use_delta_coding=False)

utils.test(true_path=path, pred_path=path_to_predictions+f_name+'_pred.npy',
    path_save=path_to_predictions+f_name+'_true_pred.png')
"""







with ECGEncoder(
    n_frames=PARAM['n_frames'],
    n_channel=PARAM['n_channels'],
    n_hidden_RNN=PARAM['n_hidden_RNN'],
    reduction_ratio=PARAM['rr'],
    use_true_inps=False,
    do_train=True) as ecg_encoder:
    
    
    ecg_encoder.train_(
        data_loader = data_loader,
        keep_prob=PARAM['keep_prob'],
        weight_decay=PARAM['weight_decay'],
        learn_rate_start=PARAM['learn_rate_start'],
        learn_rate_end=PARAM['learn_rate_end'],
        n_iter=n_iter_train,
        save_model_every_n_iter=save_model_every_n_iter,
        path_to_model=path_to_model)
    # [print(var) for var in tf.trainable_variables()]




# Predictions
path='../data/little/AAO1CMED2K865.npy'
f_name = ecg.utils.get_file_name(path)
with ECGEncoder(
    n_frames=PARAM['n_frames'],
    n_channel=PARAM['n_channels'],
    n_hidden_RNN=PARAM['n_hidden_RNN'],
    reduction_ratio=PARAM['rr'],
    use_true_inps=False,
    do_train=False) as ecg_encoder:

    ecg_encoder.predict(
        path_to_file=path,
        path_to_save=path_to_predictions+f_name+'_pred.npy',
        path_to_model=os.path.dirname(path_to_model),
        use_delta_coding=False)

utils.test(true_path=path, pred_path=path_to_predictions+f_name+'_pred.npy',
    path_save=path_to_predictions+f_name+'_pred.png')

