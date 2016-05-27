#Functions to do stuff with h5 open files


# gets the sampling frequency of a recording
def get_record_sampling_frequency(h5, recording=0):
    path = 'recordings/{0:d}'.format(recording)
    return h5[path].attrs.get('sample_rate')
