# Set of functions for appending data and reformatting the kwik file
from scipy.io import wavfile
from scipy import signal as ss
import numpy as np

# add wave stimuli
# Insert a wav file with a sound into a group


def insert_sound(kf, sound_name, sound_file_path, parent_group='/event_types/Stimulus', stim_meta_data=None, waveform_meta_data=None):
    """
    Inserts a waveform stimulus into a kwik file.

    Inserts the wav file in sound_file_path into a group under parent_group.
    It will create a group named sound_name. The group will have as default attributes:
    sampling_rate: the sampling frequency of the wav file.
    Extra parameters can be entered as a dictionary in meta_data
    The waveform is written in the dataset 'waveform' within that group.
    In principle, the name of the sound coincides with the identifier of the ../Stimulus/text dataset

    :param kf: the kwik file
    :param sound_name: name of the sound
    :param sound_file_path: path to the wavefile
    :param parent_group: stimulus group in the kwik file
    :param sitm_meta_data: dictionary of meta_data {attrib_name: value} to insert as attributes of the group.
    :yelds: inserts group sound_name with attributes and dataset with the waveform into parent_group

    """

    [s_f, data] = wavfile.read(sound_file_path)
    assert(kf[parent_group])
    sound_group = kf[parent_group].create_group(sound_name)
    waveform_group = sound_group.create_group('waveforms')
    waveform_group.attrs.create('sampling_rate', s_f, dtype='f4')

    if waveform_meta_data is not None:
        for key, val in waveform_meta_data.iteritems():
            waveform_group.attrs.create(key, val)

    if stim_meta_data is not None:
        for key, val in stim_meta_data.iteritems():
            sound_group.attrs.create(key, val)

    waveform_group.create_dataset('stimulus', data=data)


def append_stream(kf, sound_name, stream, stream_name, parent_group='/event_types/Stimulus', meta_data=None, resample=False):

    waveform_group = kf[parent_group + '/' + sound_name + '/' + 'waveforms']
    assert waveform_group
    if meta_data is not None:
        for key, val in meta_data.iteritems():
            waveform_group.attrs.create(key, val)

    if resample:
        stream = ss.resample(stream, waveform_group['stimulus'].size)

    waveform_group.create_dataset(stream_name, data=stream)


def append_table_in_stim(kf, sound_name, table, table_name, parent_group='/event_types/Stimulus', meta_data=None, table_fcn=None, **kwargs):
    stim_group = kf[parent_group + '/' + sound_name]

    if table_fcn is None:
        dset = stim_group.create_dataset(table_name, data=table)
        dict2attrs(meta_data, dset)
    else:
        table_fcn(kf, sound_name, table, table_name, parent_group, meta_data, **kwargs)


def read_sound(kf, sound_name, parent_group='/event_types/Stimulus'):
    """
    Reads the waveform of a sound stimulus and its meta_data

    :param kf: the kwik file
    :param sound_name: name of the sound
    :param parent_group: stimulus group in the kwik file
    :returns:
        data: n_samples x 1 numpy array with the waveform
        meta_data: dictionary of meta_data {attrib_name: value} to insert as attributes of the group.
    """
    waveforms_group = kf[parent_group+'/'+sound_name + '/' + 'waveforms']
    data = np.array(waveforms_group['stimulus'], dtype=waveforms_group['stimulus'].dtype)
    # read the meta_data
    meta_data = attrs2dict(waveforms_group)
    return data, meta_data


def list_stim_streams(kf, sound_name, parent_group='/event_types/Stimulus'):
    sound_group = kf[parent_group + '/' + sound_name + '/' + 'waveforms']
    return sound_group.keys()


def read_stim_stream(kf, sound_name, stream_name, parent_group='/event_types/Stimulus'):
    sound_wf_group = kf[parent_group + '/' + sound_name + '/' + 'waveforms']
    data = np.array(sound_wf_group[stream_name], dtype=sound_wf_group[stream_name].dtype)
    # read the meta_data
    meta_data = {'stim': attrs2dict(sound_wf_group), 'stream': attrs2dict(sound_wf_group[stream_name])}
    return data, meta_data


def read_stim_subtable(kf, sound_name, table_name, parent_group='/event_types/Stimulus'):
    stim_group = kf[parent_group + '/' + sound_name]
    data = np.array(stim_group[table_name], dtype=stim_group[table_name].dtype)
    # read the meta_data
    meta_data = attrs2dict(stim_group[table_name])
    return data, meta_data


def read_stim_groups(kf, parent_group='/event_types/Stimulus'):
    stim_group = kf[parent_group]
    return stim_group


# gets the sampling frequency of a recording
def get_record_sampling_frequency(h5, recording=0):
    path = 'recordings/{0:d}'.format(recording)
    return h5[path].attrs.get('sample_rate')


# List all the units in a file
def list_units(kf, group=0, sorted=True):
    # get the unit group
    qlt_path = "/channel_groups/{0:d}/clusters/main".format(group)

    g_dtype = np.int
    clu_dtype = np.int
    qlt_dtype = np.int

    clu_list = kf[qlt_path].keys()
    qlt_list = [kf["{0:s}/{1:s}".format(qlt_path, c)].attrs.get('cluster_group') for c in clu_list]
    n_spikes = len(clu_list)
    clu_dt = np.dtype([('group', g_dtype, 1), ('clu', clu_dtype, 1), ('qlt', qlt_dtype, 1)])
    clu = np.recarray(n_spikes, dtype=clu_dt)

    clu['group'] = group
    clu['clu'] = map(int, clu_list)
    clu['qlt'] = map(int, qlt_list)

    if sorted:
        clu = clu[(clu['qlt'] == 1) | (clu['qlt'] == 2)]
    return clu


# List all the stimuli in a file
def list_sound_stimuli(h5, stim_table_path='/event_types/Stimulus/text'):

    datagroup = h5[stim_table_path]
    all_stim = np.array([s for s in datagroup[:] if not is_number(s)])
    return np.unique(all_stim)


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def attrs2dict(node):
    return {key: val for key, val in node.attrs.iteritems()}


def dict2attrs(meta_dict, node):
    if meta_dict is not None:
        assert node
        for key, val in meta_dict.iteritems():
            node.attrs.create(key, val)