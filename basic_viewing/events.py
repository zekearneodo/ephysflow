# some objects to do quick stuff on events
import h5py
import numpy as np
import h5_functions as h5f


class Event:

    def __init__(self, name, h5=None):

        self.name = name
        self.start = None
        self.end = None
        self.meta = None
        self.sf = None
        self.where_event = None
        self.has_event = None
        self.get_table_function = None

        self.data = h5
        self.datagroup = None
        self.datasets = None


class Sound(Event):
    def __init__(self, name, h5=None):
        Event.__init__(self, name, h5=h5)

        self.table_columns = { #'column_name', 'dataset name'
                'name': 'text',
                'code': 'codes',
                'start': 'time_samples'}

        if self.data is not None:
            self.datagroup = h5['/event_types/Stimulus']
            self.datasets = self.datagroup.keys()

            self.get_start()
            self.get_sampling_frequency()
            self.get_has_event()

    #get the indexes in the table of occurrence of this event
    def get_idx(self):
        if self.where_event is None:
            self.where_event = np.where(self.datagroup[self.table_columns['name']][:] == self.name)

    def get_has_event(self):
        if self.has_event is None:
            self.has_event = self.datagroup[self.table_columns['name']][:] == self.name

    #get the table of events
    def get_col(self, col_name):
        if self.has_event is None:
            self.get_has_event()
        data_set = self.datagroup[self.table_columns[col_name]]
        data_type = np.dtype(data_set)
        return np.array(data_set[self.has_event], dtype=data_type)


    def get_start(self):
        self.get_col('start')
        return self.table_columns

    def get_sampling_frequency(self):
        if self.sf is not None:
            assert(self.data is not None)
            self.sf = h5f.get_record_sampling_frequency(self.data)
        return self.sf