# objects to do quick stuff with clusters
import numpy as np
import h5_functions as h5f
import kwik_functions as kf
import h5py


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
        assert(self.kwik_file is not None)

        clu_path = "/channel_groups/{0:d}/spikes/clusters/main".format(self.group)
        t_path = "/channel_groups/{0:d}/spikes/time_samples".format(self.group)
        r_path = "/channel_groups/{0:d}/spikes/recording".format(self.group)

        dtype = self.kwik_file[t_path].dtype
        self.time_samples = np.array(self.kwik_file[t_path][self.kwik_file[clu_path][:] == self.clu], dtype=np.dtype(dtype))

        dtype = self.kwik_file[r_path].dtype
        self.recordings = np.array(self.kwik_file[r_path][self.kwik_file[clu_path][:] == self.clu], dtype=np.dtype(dtype))

        return self.time_samples, self.recordings

    def get_rec_offsets(self):
        self.recording_offsets = kf.rec_start_array(self.kwik_file)
        return self.recording_offsets

    # get the quality of the unit
    def get_qlt(self):
        assert(self.kwik_file is not None)

        path = "/channel_groups/{0:d}/clusters/main/{1:d}".format(self.group, self.clu)
        self.qlt = self.kwik_file[path].attrs.get('cluster_group')

    def get_sampling_rate(self):
        assert(self.kwik_file is not None)
        self.sampling_rate = h5f.get_record_sampling_frequency(self.kwik_file)

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
            assert(starts.size==recs.size)
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
            rec_time_samples = self.time_samples[self.recordings==rec]
            for start in starts[recs==rec]:
                start-=start_rec_offsets[rec_list==rec]
                end = np.int(start + span_samples)
                where = (rec_time_samples[:] >= start) & (rec_time_samples[:] <= end)
                n = np.sum(where)
                raster[i_trial, :n] = rec_time_samples[where] - start
                if return_ms:
                    raster[i_trial, :n] = np.round(raster[i_trial, :n] * 1000. / self.sampling_rate)
                i_trial+=1
        return raster
