import logging
import os
import socket
import sys

import numpy as np
import scipy as sp
import scipy.signal as sg

# Check which computer to decide where the things are mounted
comp_name = socket.gethostname()

if 'txori' in comp_name or 'passaro' in comp_name or 'lintu' in comp_name:
    repos_folder = os.path.abspath('/mnt/cube/earneodo/repos')
    experiment_folder = os.path.join('/mnt/cube/earneodo/bci_zf/')

sys.path.append(os.path.join(repos_folder, 'ephysflow'))
sys.path.append(os.path.join(repos_folder, 'swissknife'))

from basic_viewing.structure.core import basic_plot as bp

from basic_viewing.structure.core import h5_functions as kwdf

from swissknife.streamtools import streams as st, temporal as tp
from swissknife.streamtools import spectral as sp

from swissknife.decoder import linear as ld
from swissknife.bci import synthetic as syn

logger = logging.getLogger('spike_detect_aux')
logger.info('Computer: {}'.format(comp_name))


def list_sum(a_list):
    s = a_list.pop(-1)
    for new_s in a_list:
        s += new_s
    return s


def band_pass_filter(chunk, hp_b, hp_a, lp_b, lp_a):
    chunk_hi = sg.filtfilt(hp_b, hp_a, chunk)
    chunk_filt = sg.filtfilt(lp_b, lp_a, chunk_hi)
    return chunk_filt


def filter_rms(x, filter_pars):
    return st.rms(sp.apply_butter_bandpass(x, filter_pars))


def collect_frames(starts, span, s_f, kwd_file, recs_list, chan_list):
    frames = []
    logger.info('Collecting {} frames...'.format(starts.size))
    for i_start, start in enumerate(starts):
        if i_start % 10 == 0:
            logger.info("Frame {} ...".format(i_start))
        rec = recs_list[i_start]
        one_frame = st.Chunk(st.H5Data(kwdf.get_data_set(kwd_file, rec),
                                       s_f,
                                       dtype=np.float),
                             np.array(chan_list),
                             [start, start + span])
        frames.append(one_frame)
    return frames


def spikes_array(chunk, thresholds, min_dist=10):
    # logger.info('Getting spikes from chunk with data sized {}'.format(chunk.data.shape))
    spk_lst = tp.find_spikes(chunk.data, thresholds, min_dist=min_dist)
    spk_arr = np.zeros_like(chunk.data)
    assert (len(spk_lst) == spk_arr.shape[1])
    for ch in range(len(spk_lst)):
        spk_arr[spk_lst[ch], ch] = 1
    return spk_arr


def collect_all_spk_arr(frames_list, thresholds, min_dist=10):
    return np.stack([spikes_array(fr, thresholds, min_dist=min_dist) for fr in frames_list], axis=0)


def find_silent(sup_vec):
    silent_list = np.array([any(~(sup_vec[i, :, :].any(axis=0))) for i in range(sup_vec.shape[0])])
    return silent_list


def support_vector_from_raw(starts, recs, len_samples, channels, thresholds, kwd_file,
                            bin_size=10, s_f=30000, history_bins=1, no_silent=True):
    bin_size_samples = int(bin_size * s_f / 1000.)
    len_bin = int(len_samples / bin_size_samples)
    len_ms = int(len_bin * bin_size)
    history_samples = history_bins * bin_size_samples
    span_ms = len_ms + bin_size * history_bins
    span_samples = int(span_ms * s_f / 1000.)

    logger.info('Creating support vector {0} chans, {1} trials'.format(channels.size, starts.size))
    # logger.info('span_ms = {}'.format(span_ms))
    # logger.info('{} units'.format(len(all_units)))

    all_frames = collect_frames(starts - history_samples,
                                span_samples,
                                s_f,
                                kwd_file,
                                recs,
                                channels)

    filter_band = [500, 10000]
    filter_pars = sp.make_butter_bandpass(s_f, filter_band[0], filter_band[1])
    [fr.apply_filter(sp.apply_butter_bandpass, filter_pars) for fr in all_frames]
    all_spk_arr = collect_all_spk_arr(all_frames, thresholds)

    rst_sv = np.stack([bp.col_binned(all_spk_arr[t, :, :].T, bin_size_samples) for t in range(all_spk_arr.shape[0])],
                      axis=2)

    if no_silent:
        good_chans = ~find_silent(rst_sv)
    else:
        good_chans = np.arange(channels.size)

    return rst_sv[good_chans, :, :], good_chans


# INTO THE WILD FITTING NOW
def transform_env(x):
    x[x < 0] = 0
    return x


def transform_alpha(x):
    x[x < 0.15] = 0
    x[x >= .15] = .3
    return x


def transform_beta(x):
    x[x < 0] = 0
    return x


def mu_transform_beta(x):
    x[x < 0] = 0
    return syn.np_mulog_inv(x, 256)


def linear_fit(channels, thresholds, kwd_file, trial_starts, trial_recs, par_stream,
               bin_size=10,
               history_bins=15,
               s_f=30000):
    logger.info('Fitting a kernel')
    bin_size_samples = int(bin_size * s_f / 1000.)
    len_samples = par_stream.shape[0]
    model_pars = bp.col_binned(np.array([par_stream]), bin_size_samples) / bin_size_samples

    s_v, chans = support_vector_from_raw(trial_starts, trial_recs, len_samples, channels, thresholds, kwd_file,
                                         bin_size=bin_size, s_f=s_f, history_bins=history_bins + 1, no_silent=True)

    target = np.tile(model_pars, trial_starts.size).reshape(trial_starts.size, -1)
    logger.info('sv shape {0}, target shape {1}'.format(s_v.shape, target.shape))
    return ld.fit_kernel(s_v, target, history_bins + 1), chans


def linear_predict(channels, thresholds, kwd_file, trial_starts, trial_recs, len_samples, kern,
                   bin_size=10,
                   history_bins=15,
                   s_f=30000,
                   no_silent=False):
    # kern.flatten()
    logger.info('k shape {}'.format(kern.shape))
    logger.info('Convolving a kernel')
    logger.info('Channels are {}'.format(channels.size))
    s_v, chan_list = support_vector_from_raw(trial_starts, trial_recs, len_samples, channels, thresholds, kwd_file,
                                             bin_size=bin_size, s_f=s_f, history_bins=history_bins + 1,
                                             no_silent=no_silent)

    logger.info('kernel shape {0}, sv_shape {1}, len_samples {2}'.format(kern.shape, s_v.shape, len_samples))
    return ld.kernel_predict(s_v, kern)


def test_fit(channels, thresholds, kwd_file, starts, recs, trials_fit, trials_test,
             par_stream,
             bin_size=10,
             history_bins=15,
             s_f=30000,
             nonlinear_fun=lambda x: x):
    fitted_kernel, fitted_chans = linear_fit(channels,
                                             thresholds,
                                             kwd_file,
                                             starts[trials_fit],
                                             recs[trials_fit],
                                             par_stream,
                                             bin_size=bin_size,
                                             history_bins=history_bins,
                                             s_f=s_f)

    par_predict = linear_predict(channels[fitted_chans], thresholds[fitted_chans],
                                 kwd_file,
                                 starts[trials_test],
                                 recs[trials_test],
                                 par_stream.size,
                                 fitted_kernel,
                                 bin_size=bin_size,
                                 history_bins=history_bins,
                                 s_f=30000,
                                 no_silent=False)

    bin_size_samples = int(bin_size * s_f / 1000.)
    binned_pars = bp.col_binned(np.array([par_stream]), bin_size_samples) / bin_size_samples
    target = np.tile(binned_pars, trials_test.size).reshape(trials_test.size, -1)

    kernel_predict = nonlinear_fun(par_predict)
    assert (trials_fit.size > trials_test.size)
    residue = np.linalg.norm(kernel_predict - target) / np.linalg.norm(target)
    return fitted_kernel, kernel_predict, residue, target, fitted_chans


def altogether_test(alpha, beta, env,
                        channels,
                        thresholds,
                        kwd_file,
                        starts,
                        recs,
                        trials_train,
                        trials_test,
                        bin_size=7,
                        history_bins=15,
                        s_f=30000,
                        nl_alpha=transform_alpha,
                        nl_beta=transform_beta,
                        nl_env=transform_env):

    logger.info('Testing for all pars with bin_size={0}, history_size={1}:'.format(bin_size, history_bins))
    all_tests = [[] for i in range(5)]
    for par, nl in zip([alpha, beta, env], [nl_alpha, nl_beta, nl_env]):
        logger.info('Testing fit for a parameter')
        tested = test_fit(channels,
                          thresholds,
                          kwd_file,
                          starts,
                          recs,
                          trials_train,
                          trials_test,
                          par,
                          bin_size=bin_size,
                          history_bins=history_bins,
                          s_f=s_f,
                          nonlinear_fun=nl
                          )
        for i, test_res in enumerate(tested):
            all_tests[i].append(test_res)

    return [np.stack(t, axis=0) for t in all_tests]