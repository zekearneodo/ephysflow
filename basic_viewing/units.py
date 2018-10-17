# objects to do quick stuff with clusters
from __future__ import division

import numpy as np
import logging

from swissknife.bci.core.file import h5_functions as h5f
from swissknife.bci.core import kwik_functions as h5f

from swissknife.bci.core import basic_plot as bp

logger = logging.getLogger('swissknife.bci.core.units')


class Unit:
    def __init__(self, clu, group=0, kwik_file=None, sort=0):

        self.clu = clu
        self.group = group

        self.kwik_file = kwik_file
        self.sort = sort

        self.id = None
        self.metrics = None

        self.qlt = None
        self.time_samples = None
        self.recordings = None
        self.recording_offsets = None
        self.sampling_rate = None

        if kwik_file is not None:
            self.get_sampling_rate()
            self.get_qlt()
            self.get_time_stamps()
            self.get_rec_offsets()

    # get time stamps of spiking events (in samples)
    def get_time_stamps(self):
        assert (self.kwik_file is not None)

        clu_path = "/channel_groups/{0:d}/spikes/clusters/main".format(self.group)
        t_path = "/channel_groups/{0:d}/spikes/time_samples".format(self.group)
        r_path = "/channel_groups/{0:d}/spikes/recording".format(self.group)

        dtype = self.kwik_file[t_path].dtype
        time_samples = np.array(self.kwik_file[t_path][self.kwik_file[clu_path][:] == self.clu],
                                dtype=np.dtype(dtype))

        dtype = self.kwik_file[r_path].dtype
        recordings = np.array(self.kwik_file[r_path][self.kwik_file[clu_path][:] == self.clu],
                              dtype=np.dtype(dtype))

        # patch for a random kilosort error that throws a random 0 for a time_stamp
        self.time_samples = time_samples[time_samples > 0]
        self.recordings = recordings[time_samples > 0]
        return self.time_samples, self.recordings

    def get_rec_offsets(self):
        self.recording_offsets = kf.rec_start_array(self.kwik_file)
        return self.recording_offsets

    # get the quality of the unit
    def get_qlt(self):
        assert (self.kwik_file is not None)

        path = "/channel_groups/{0:d}/clusters/main/{1:d}".format(self.group, self.clu)
        self.qlt = self.kwik_file[path].attrs.get('cluster_group')

    def get_sampling_rate(self):
        assert (self.kwik_file is not None)
        self.sampling_rate = h5f.get_record_sampling_frequency(self.kwik_file)
        return self.sampling_rate

    def get_raster(self, starts, span, span_is_ms=True, return_ms=None, recs=None):
        """
        :param starts: start points of each event (in samples, absolute unles recs is provided)
        :param span: span of the raster (in samples or ms, depending on value of span_is_ms)
        :param span_is_ms: whether the span of the raster is given in samples or ms, default is ms
        :param return_ms: whether to return the raster in ms units, default is to do the units set by span_is_ms
        :param recs: if recs is provided, starts are referred to the beginning of each rec
                     otherwise, the method will identify which rec each start belongs to and offset accordingly
        :return: n x span
        """

        if recs is None:
            recs = kf.get_corresponding_rec(self.kwik_file, starts)
            start_rec_offsets = kf.rec_start_array(self.kwik_file)
            rec_list = kf.get_rec_list(self.kwik_file)
        else:
            assert (starts.size == recs.size)
            rec_list = kf.get_rec_list(self.kwik_file)
            start_rec_offsets = np.zeros_like(rec_list)

        return_ms = span_is_ms if return_ms is None else return_ms
        span_samples = np.int(span * self.sampling_rate * 0.001) if span_is_ms else span
        span_ms = span if span_is_ms else np.int(span * 1000. / self.sampling_rate)

        rows = starts.shape[0]
        cols = span_ms if return_ms else span_samples
        raster = np.empty((rows, cols), dtype=np.float64)
        raster[:] = np.nan

        # do the raster in samples
        i_trial = 0
        for rec in np.unique(recs):
            rec_time_samples = self.time_samples[self.recordings == rec]
            for start in starts[recs == rec]:
                start -= start_rec_offsets[rec_list == rec]
                end = np.int(start + span_samples)
                where = (rec_time_samples[:] >= start) & (rec_time_samples[:] <= end)
                n = np.sum(where)
                raster[i_trial, :n] = rec_time_samples[where] - start
                if return_ms:
                    raster[i_trial, :n] = np.round(raster[i_trial, :n] * 1000. / self.sampling_rate)
                i_trial += 1
        return raster


class kiloUnit(Unit):
    def __init__(self, clu, group=0, kwik_file=None, sort=0):
        super(Unit, self).__init__(clu, group=group, kwik_file=kwik_file, sort=sort)


def support_vector(starts, len_samples, all_units, bin_size=10, s_f=30000, history_bins=1, no_silent=True):
    """
    :param starts: list or np array of starting points
    :param len_samples: length in samples of the 'trial'
    :param all_units: list of Unit objects (as in units.py)
    :param bin_size: size of the bin for the spike count
    :param history_bins: number of bins previous to starting points to include
    :param no_silent: exclude units that don't spike (to prevent singular support arrays)
    :return: np array [n_bins, n_units, n_trials] (compatible with other features sup vecs)
    """
    bin_size_samples = int(bin_size * s_f / 1000.)
    len_bin = int(len_samples / bin_size_samples)
    len_ms = int(len_bin * bin_size)

    history_samples = history_bins * bin_size_samples

    span_ms = len_ms + bin_size * history_bins
    # logger.info('span_ms = {}'.format(span_ms))
    sup_vec = []
    sup_vec_units = []
    # logger.info('{} units'.format(len(all_units)))
    for i, a_unit in enumerate(all_units):
        raster = a_unit.get_raster(starts - history_samples,
                                   span_ms,
                                   span_is_ms=True,
                                   return_ms=True)
        sparse_raster = bp.col_binned(bp.sparse_raster(raster), bin_size)

        if no_silent and not sparse_raster.any():
            logger.warn('Watch out, found lazy unit')
            pass
        else:
            sup_vec.append(sparse_raster.T)
            sup_vec_units.append(a_unit)
    # logger.info('sparse raster shape = {}'.format(sparse_raster.shape))
    # return sup_vec
    return np.stack(sup_vec, axis=0), sup_vec_units


def filter_unit_list(in_list, filter_func, *args, **kwargs):
    return [unit for unit in in_list if filter_func(unit, *args, **kwargs)]


def no_silent_filter(a_unit, starts, len_samples, bin_size=10, s_f=30000, history_bins=1):
    """
    :param starts: list or np array of starting points
    :param len_samples: length in samples of the 'trial'
    :param a_unit: one Unit objects (as in units.py)
    :param bin_size: size of the bin for the spike count
    :param history_bins: number of bins previous to starting points to include
    :return: True if the unit has at leas one spike in the raster
    """
    bin_size_samples = int(bin_size * s_f / 1000.)
    len_bin = int(len_samples / bin_size_samples)
    len_ms = int(len_bin * bin_size)
    history_samples = history_bins * bin_size_samples
    span_ms = len_ms + bin_size * history_bins

    raster = a_unit.get_raster(starts - history_samples,
                               span_ms,
                               span_is_ms=True,
                               return_ms=True)

    sparse_raster = bp.col_binned(bp.sparse_raster(raster), bin_size)

    return not any(~sparse_raster.any(axis=1))


def no_singularity_filter(a_unit, starts, len_samples, bin_size=10, s_f=30000, history_bins=1):
    """
    :param starts: list or np array of starting points
    :param len_samples: length in samples of the 'trial'
    :param a_unit: one Unit objects (as in units.py)
    :param bin_size: size of the bin for the spike count
    :param history_bins: number of bins previous to starting points to include
    :return: True if the unit has at leas one spike in the raster
    """
    bin_size_samples = int(bin_size * s_f / 1000.)
    len_bin = int(len_samples / bin_size_samples)
    len_ms = int(len_bin * bin_size)
    history_samples = history_bins * bin_size_samples
    span_ms = len_ms + bin_size * history_bins

    raster = a_unit.get_raster(starts - history_samples,
                               span_ms,
                               span_is_ms=True,
                               return_ms=True)

    sparse_raster = bp.col_binned(bp.sparse_raster(raster), bin_size)

    return (sparse_raster.any())