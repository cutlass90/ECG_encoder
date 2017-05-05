import os
import sys
import argparse
import importlib
from random import shuffle

import numpy as np
import matplotlib.pyplot as plt
import tqdm
from sklearn.cluster import KMeans

from ecg.utils import tools
from ecg.utils.diseases import holter_diseases_with_noise as new_diseases

cache_path = 'cluster_cache'
plot_save_path = 'clustering_plots'
channel_names = ['ES', 'AS', 'AI']
sample_rate = 175
# how many seconds of ECG to plot on both sides
seconds = 15

def get_file_pointers_for_cluster_centers(cluster_labels, clustering_data, cluster_indexes):
    file_pointers = list()
    for cluster_idx in cluster_indexes:
        #find z and file pointers for current cluster
        idx = np.argwhere(np.array(cluster_labels) == cluster_idx).ravel()
        cluster_v = np.vstack([data['state'] for data in clustering_data])[idx]
        cluster_pointers = np.vstack([np.hstack(data['fp']) 
                            for data in clustering_data])[idx]
        cluster_center = np.mean(cluster_v, 0)

        def find_central_z_idx(array, value):
            idx = (np.sum(np.square(array-value), 1)).argmin()
            return idx

        #find file pointer which corresponds to cluster center
        idx = find_central_z_idx(cluster_v, cluster_center)
        file_pointer = cluster_pointers[idx]
        file_pointers.append(file_pointer)
    return file_pointers


def display_dists_and_strength(snn, disease_labels, snn_str, snn_dists):

    pvc = new_diseases.index('PVC')
    pvc_idx = np.where(disease_labels[:, pvc])[0]

    top_str_idx = snn[np.arange(len(snn)), np.argmax(snn_str, 1)]
    top_str_idx_pvc = top_str_idx[pvc_idx]

    top_dist_idx = snn[np.arange(len(snn)), np.argmin(snn_dists, 1)]
    top_dist_idx_pvc = top_dist_idx[pvc_idx]


    print('Top str snn for pvc beats:')
    for top_label in disease_labels[top_str_idx_pvc]:
        print(new_diseases[np.where(top_label != 0)[0][0]])
    print()

    print('Top distance snn for pvc beats:')
    for top_label in disease_labels[top_dist_idx_pvc]:
        print(new_diseases[np.where(top_label != 0)[0][0]])
    print()

    unique_snn = np.unique(snn[pvc_idx])
    print('PVC indexes: {}, unique enterings in PVC SNN list: {}, intersection: {}'.format(
        len(pvc_idx), len(unique_snn), len(np.intersect1d(pvc_idx, unique_snn))))


def get_cluster_labels(clustering_data, n_clusters, use_snn_clustering=False):
    print('Starting clusterting with {} clusters.'.format(n_clusters))

    hidden_states = np.vstack([d['state'] for d in clustering_data]).astype(np.float32)
    disease_labels = np.vstack([d['label'] for d in clustering_data]).astype(np.int32)

    if use_snn_clustering:
        print('Clustering algorithm: SNN')
        cluster_labels, snn, snn_str, snn_dists = tools.cluster_snn(
                                hidden_states, n_clusters, dist_func='l2norm')
    else:
        print('Clustering algorithm: KMeans')
        model = KMeans(n_clusters=n_clusters, max_iter=1000)
        cluster_labels = model.fit_predict(hidden_states)
    print('Clustering finished.')
    cluster_idx = set(cluster_labels)
    return cluster_labels, cluster_idx

def print_clustering_stats(cluster_labels, clustering_data):

    clustering_data = np.vstack([d['label'] for d in clustering_data])

    print('\nSummary disease count:')
    for i, disease_count in enumerate(np.sum(clustering_data, 0)):
        if disease_count:
            print('Disease: {}, count: {}, ratio: {}'.format(
                new_diseases[i], disease_count, disease_count/len(clustering_data)))


    for label in set(cluster_labels):
        cluster_data = clustering_data[cluster_labels == label]
        summary_labels = np.sum(cluster_data, 0)
        print('\nCluster {} with size {}.'.format(label, len(cluster_data)))
        print('Diseases in cluster:')
        for i, disease_count in enumerate(summary_labels):
            if disease_count:
                print('{}, count: {}, ratio: {}'.format(
                    new_diseases[i], disease_count, disease_count/len(cluster_data)))

def plot_beats(file_pointers, save_path, caching=True, skip_prob=0):
    print('Creating plots...')
    tools.maybe_create_dirs(save_path)
    if caching:
        cache = {}
    for pointer in tqdm.tqdm(file_pointers, ncols=80):
        if np.random.uniform() < skip_prob:
            continue
        f, beat_idx = pointer
        beat_idx = int(beat_idx)
        
        if caching:
            if f in cache:
                data = cache[f]
            else:
                data = np.load(f).item()
                cache[f] = data
        else:
            data = np.load(f).item()
        
        channels = tools.get_channels(data)
        beats = data['beats'][1:-1]
        cluster_beat = beats[beat_idx]
        l, r = (cluster_beat-sample_rate*seconds, cluster_beat+sample_rate*seconds)
        if l < 0 or r > len(channels[0]):
            continue
        channels = [ch[l:r] for ch in channels]
        fig = plt.figure(figsize=(15, len(channels)*3))
        channel_name = iter(channel_names)
        ylims = (-0.6, 0.6)
        for i, ch in enumerate(channels):
            t = range(l, r)
            ax = fig.add_subplot(len(channels), 1, i+1)
            ax.plot(t, ch, lw=1.0, c='b', alpha=0.7)                                        
            ax.set_ylim(ylims)
            ax.set_xlim(l, r)
            ax.set_title(next(channel_name))
            ax.axvspan(cluster_beat-70, cluster_beat+70, facecolor='g', alpha=0.2)
            plt.setp(ax.get_xticklabels(), visible=False)
            plt.setp(ax.get_yticklabels(), visible=False)

        fig.savefig(
            os.path.join(
                   save_path,
                   tools.get_file_name(f) + '_{}.png'.format(cluster_beat)),
            tight_layout=True)
        plt.close(fig)
    
    print('Finished plotting.')

def plot_clusters(clustering_data, cluster_labels, save_path):

    clustering_data = np.asarray(clustering_data)
    cluster_data = {}
    for label in set(cluster_labels):
        cluster_data[label] = clustering_data[cluster_labels == label]
    for label, data in cluster_data.items():
        skip_prob = 0#min(100*len(data)/len(clustering_data), 0.999)
        print('\nPlotting cluster with label {}, with total size of {}. Skip prob: {}.'.format(
            label, len(data), skip_prob))
        plot_beats(
                   file_pointers=np.vstack([d['fp'] for d in data]),
                   save_path=os.path.join(save_path, '{}_{}total'.format(label, len(data))),
                   caching=True,
                   skip_prob=skip_prob)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
                        '--n_clusters', type=int,
                        default=10, help='number of clusters')
    parser.add_argument(
                        '--save_dir', type=str,
                        help='dir to save plots in', required=True)
    parser.add_argument(
                    '--use_snn', default=False,
                     dest='use_snn', action='store_true')
    args = parser.parse_args()

    def create_clustering_data():
        # should return a list of dicts with the following keys:
        # `fp` - tuple, pointer to file (file_name, beat_idx)
        # `state` - 1-d numpy array which represents embedding (z-code, whatever) 
        # `label` - 1-d numpy array which represents diseases associated with state
        from ecg_encoder_parameters import parameters as PARAM
        import ecg_encoder_tools as utils
        from ecg_encoder import ECGEncoder
        import ecg

        path_to_model = 'models/ecg_encoder'

        gen_params = dict(n_frames = 1,
                overlap = PARAM['n_frames']-1,
                get_data = not(PARAM['use_delta_coding']),
                get_delta_coded_data = PARAM['use_delta_coding'],
                get_events = False,
                rr = PARAM['rr'])

        # Get Z-code
        path = '../data/little/AAO1CMED2K865.npy'
        with ECGEncoder(
            n_frames=PARAM['n_frames'],
            n_channel=PARAM['n_channels'],
            n_hidden_RNN=PARAM['n_hidden_RNN'],
            reduction_ratio=PARAM['rr'],
            frame_weights=PARAM['frame_weights'],
            do_train=False) as ecg_encoder:

            Z = ecg_encoder.get_Z(
                path_to_file=path,
                path_to_save=None,
                path_to_model=os.path.dirname(path_to_model),
                use_delta_coding=False)
        
        data = np.load(path).item()
        names = data['disease_name']
        events = tools.remove_redundant_events(data['events'], names, new_diseases)

        list_of_samples = []
        for i in range(data['beats'].shape[0]-2):
            sample = {}
            sample['fp'] = (path, i)
            sample['label'] = events[i,:]
            sample['state'] = Z[i,:]
            list_of_samples.append(sample)
        # print(len(list_of_samples))
        # print(list_of_samples[0])
        # raise TypeError
        return list_of_samples

    clustering_data = tools.run_with_caching(create_clustering_data, cache_path)
    print('Clustering data size: {}'.format(len(clustering_data)))
    n_clusters = args.n_clusters
    cluster_labels, cluster_idx = get_cluster_labels(clustering_data, n_clusters, args.use_snn)
    print_clustering_stats(cluster_labels, clustering_data)
    plot_clusters(clustering_data, cluster_labels, args.save_dir)
    # file_pointers = get_file_pointers_for_cluster_centers(cluster_labels, clustering_data, cluster_idx)
    # plot_beats(file_pointers, args.save_dir)