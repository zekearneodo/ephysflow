# some objects to do quick stuff on events
import h5py
import numpy as np
from phy_tools import kwik_functions as kwf


class Event:

    name = None
    start = None
    end = None
    meta = None
    sampling_rate = None
    where_event = None
    has_event = None
    get_table_function = None

    data = None
    datagroup = None
    datasets = None

    def __init__(self, name, h5=None):
        self.name = name
        self.data = h5


class Sound(Event):
    def __init__(self, name, h5=None):
        Event.__init__(self, name, h5=h5)

        self.table_columns = { #'column_name', 'dataset name'
                'name': 'text',
                'code': 'codes',
                'start': 'time_samples'}

        if self.data is not None:
            self.datagroup_path = '/event_types/Stimulus'
            self.datagroup = h5[self.datagroup_path]
            self.datasets = self.datagroup.keys()

            self.get_start()
            self.get_sampling_rate()
            self.get_where_event()

    # get the indexes in the table of occurrence of this event
    def get_idx(self):
        if self.where_event is None:
            self.where_event = np.where(self.datagroup[self.table_columns['name']][:] == self.name)

    # Query where this event happened
    def get_where_event(self):
        if self.has_event is None:
            self.has_event = self.datagroup[self.table_columns['name']][:] == self.name

    # get the table of events
    def get_col(self, col_name):
        if self.has_event is None:
            self.get_where_event()
        data_set = self.datagroup[self.table_columns[col_name]]
        data_type = np.dtype(data_set)
        return np.array(data_set[self.has_event], dtype=data_type)

    # get starting samples of events
    def get_start(self):
        if self.start is None:
            self.start = self.get_col('start')
        return self.get_col('start')

    def get_meta(self):
        return kwf.attrs2dict(self.data[self.name])

    def get_sampling_rate(self):
        if self.sampling_rate is None:
            self.sampling_rate = kwf.get_record_sampling_frequency(self.data)
        return self.sampling_rate

    def get_waveform(self, stream='stimulus'):
        return kwf.read_stim_stream(self.data, self.name, stream_name=stream, parent_group=self.datagroup_path)

    def list_waveforms(self):
        return kwf.list_stim_streams(self.data, self.name, parent_group=self.datagroup_path)

    def get_syllables(self, table_name='syllables'):
        raw_table = np.array(kwf.read_stim_subtable(self.data, self.name, table_name, parent_group=self.datagroup_path))
        return raw_table
