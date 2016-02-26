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

    def get_time_stamps(self):
        assert(self.data is not None)

        clu_path = "/channel_groups/{0:d}/spikes/clusters/main".format(self.group)
        t_path = "/channel_groups/{0:d}/spikes/time_samples".format(self.group)

        dtype = self.data[t_path].dtype

        self.time_samples = np.array(self.data[t_path][self.data[clu_path][:] == self.clu], dtype=np.dtype(dtype))

    def get_qlt(self):
        assert(self.data is not None)

        path = "/channel_groups/{0:d}/clusters/main/{1:d}".format(self.group, self.clu)
        self.qlt = self.data[path].attrs.get('cluster_group')

    def get_sampling_frequency(self):
        assert(self.data is not None)
        self.s_f = h5f.get_record_sampling_frequency(self.data)

    def get_raster(self, starts, span, span_is_ms=True):
        """
        :param starts: start points of each event (in samples)
        :param span: span of the raster (in samples or ms, depending on value of span_is_ms)
        :param span_is_ms: whether the span of the raster is given in samples or ms, default is ms
        :return:
        """
        rows = starts.shape[0]
        span_samples = np.int(span * self.s_f * 0.001) if span_is_ms else span

        raster = np.empty((rows, span_samples))
        raster[:] = np.nan

        for i, start in enumerate(starts):
            end = np.int(start + span_samples)
            where = (self.time_samples[:] >= start) & (self.time_samples[:] <= end)
            n = np.sum(where)
            raster[i, :n] = self.time_samples[where] - start

        return raster
