# objects to do quick stuff with clusters
import numpy as np
import h5_functions as h5f

import h5py


class Unit:

    def __init__(self, clu, group=0, h5=None, sort=0):

        self.clu = clu
        self.group = group

        self.data = h5
        self.sort = sort

        self.id = None
        self.metrics = None

        self.qlt = None
        self.time_samples = None
        self.s_f = None

        if h5 is not None:
            self.get_sampling_frequency()
            self.get_qlt()
            self.get_time_stamps()

    # get time stamps of spiking events (in samples)
    def get_time_stamps(self):
        assert(self.data is not None)

        clu_path = "/channel_groups/{0:d}/spikes/clusters/main".format(self.group)
        t_path = "/channel_groups/{0:d}/spikes/time_samples".format(self.group)

        dtype = self.data[t_path].dtype

        self.time_samples = np.array(self.data[t_path][self.data[clu_path][:] == self.clu], dtype=np.dtype(dtype))

    # get the quality of the unit
    def get_qlt(self):
        assert(self.data is not None)

        path = "/channel_groups/{0:d}/clusters/main/{1:d}".format(self.group, self.clu)
        self.qlt = self.data[path].attrs.get('cluster_group')

    def get_sampling_frequency(self):
        assert(self.data is not None)
        self.s_f = h5f.get_record_sampling_frequency(self.data)

    def get_raster(self, starts, span, span_is_ms=True, return_ms=None):
        """
        :param starts: start points of each event (in samples)
        :param span: span of the raster (in samples or ms, depending on value of span_is_ms)
        :param span_is_ms: whether the span of the raster is given in samples or ms, default is ms
        :param return_ms: whether to return the raster in ms units, default is to do the units set by span_is_ms
        :return: n x span array
        """
        return_ms = span_is_ms if return_ms is None else return_ms
        span_samples = np.int(span * self.s_f * 0.001) if span_is_ms else span
        span_ms = span if span_is_ms else np.int(span * 1000./self.s_f)

        rows = starts.shape[0]
        cols = span_ms if return_ms else span_samples
        raster = np.empty((rows, cols), dtype=np.float64)
        raster[:] = np.nan

        # do the raster in samples
        for i, start in enumerate(starts):
            end = np.int(start + span_samples)
            where = (self.time_samples[:] >= start) & (self.time_samples[:] <= end)
            n = np.sum(where)
            raster[i, :n] = self.time_samples[where] - start
            if return_ms:
                raster[i, :n] = np.round(raster[i, :n] * 1000./self.s_f)
        return raster
