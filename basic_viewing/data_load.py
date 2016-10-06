__author__ = 'zeke'
# functions for getting units
import numpy as np
import scipy.io as sio
import scipy as sp
import os


# Load a baseline sniff file
def load_sniff_base(mat_file_path, as_dict=True):
    assert (os.path.isfile(mat_file_path))
    # print sio.whosmat(mat_file)
    base_data = sio.loadmat(mat_file_path, struct_as_record=False, squeeze_me=True)

    # print type(cell_data['raster'])
    # print len(cell_data['raster'])
    # num_recs = len(cell_data['raster']) #num of recs the cell spans

    if type(base_data['trialsBase']) == np.ndarray:
        trialsBase = base_data['trialsBase']
    else:
        # there is only one trial!
        # this is bad; but I will debug later.
        # for now just make a list with the only trial
        trialsBase = np.array([base_data['trialsBase']])

    # print trialsBase[0].sniffFlow
    num_trials = trialsBase.shape[0]
    num_tpoints = trialsBase[0].sniffFlow.shape[0]
    # print num_trials

    sniff_flow = np.empty([num_tpoints, num_trials], dtype=int)
    sniff_phase = np.empty([num_tpoints, num_trials], dtype=int)
    sniff_start = np.empty([num_trials, 1], dtype=int)
    # print sniff_flow.shape

    for i in range(num_trials):
        # print i
        tb = trialsBase[i]
        sniff_flow[:, i] = tb.sniffFlow
        sniff_phase[:, i] = tb.sniffPhase[0: (num_tpoints)]
        sniff_start[i] = tb.start

    # print tb.sniffFlow.shape
    # print sniff_start
    # print tb['start'][0][0]
    # print tb['trialUId'][0]

    # get the rec_id from the file_name
    rec_id = os.path.split(mat_file_path)[-1].split('trialsBase.mat')[0][:-1]

    sniffs = {'flow': sniff_flow,
              'phase': sniff_phase,
              'start': sniff_start,
              'trial_uid': [str(x.trialUId) for x in trialsBase],
              'rec_id': rec_id
              }

    if as_dict:
        return {rec_id: sniffs}
    else:
        return sniffs


# Load a baseline raster file
def get_baseline(spikesBase):
    num_trials = spikesBase.spikes.shape[0]
    # print 'numtrials ' + str(num_trials)
    num_tpoints = spikesBase.spikes.shape[0]
    # print 'numpoints ' + str(num_tpoints)

    # sr_spikes = np.array(spikesBase.spikes, dtype=np.float)
    # sr_t0     = np.array(spikesBase.t0, dtype = np.float)

    base_spikes = {'spikes': np.array(spikesBase.spikes, dtype=np.float),
                   't_0': np.array(spikesBase.t0, dtype=np.float),
                   't_1': spikesBase.t1,
                   't_2': spikesBase.t2,
                   'mouse ': str(spikesBase.mouse),
                   'sess': spikesBase.sess,
                   'rec': str(spikesBase.rec),
                   'u_id': str(spikesBase.uid),
                   'id': str(spikesBase.cellId),
                   'rec_id': str(spikesBase.mouse) + '_' + str(spikesBase.sess).zfill(3) + '_' + str(spikesBase.rec)
                   }

    return base_spikes


# Load a baseline raster file
def load_baseline(mat_file_path, as_dict=True):
    assert (os.path.isfile(mat_file_path))
    base_data = sio.loadmat(mat_file_path, struct_as_record=False, squeeze_me=True)

    baselines = {}
    if type(base_data['spikesBase']) == np.ndarray:
        for base in base_data['spikesBase']:
            # print rec
            baseline = get_baseline(base)
            baselines[baseline['id']] = baseline
    else:
        baseline = get_baseline(base_data['spikesBase'])
        baselines[baseline['id']] = baseline

    if as_dict:
        return baselines
    else:
        baselines_list = [baselines[a_key] for a_key in baselines.keys()]
        return baselines_list


# Load a sniff file into a rec array
def load_sniffs(mat_file_path, as_dict=True):
    # print mat_file_path
    assert (os.path.isfile(mat_file_path))
    struct_name = os.path.split(mat_file_path)[-1].split('.')[0].split('_')[-1]
    sniff_data = sio.loadmat(mat_file_path, struct_as_record=False, squeeze_me=True)

    if type(sniff_data[struct_name]) == np.ndarray:
        # if there are many sniffs
        num_sniffs = sniff_data[struct_name].shape[0]
        sniffs = np.zeros((num_sniffs,), dtype=np.dtype([('flow', np.ndarray),
                                                         ('t_0', np.int),
                                                         ('t_zer', np.ndarray),
                                                         ('t_zer_fit', np.ndarray),
                                                         ('inh_len', np.int),
                                                         ('exh_len', np.int)]))
        i_sniff = 0
        for sniff_struct in sniff_data[struct_name]:
            sniffs[i_sniff]['flow'] = np.array(sniff_struct.waveform, dtype=np.float)
            sniffs[i_sniff]['t_0'] = sniff_struct.t0
            sniffs[i_sniff]['t_zer'] = np.array(sniff_struct.t_zer, dtype=np.int)
            sniffs[i_sniff]['t_zer_fit'] = np.array(sniff_struct.t_zer_fit, dtype=np.float)
            sniffs[i_sniff]['inh_len'] = sniff_struct.t_zer_fit[1] - sniff_struct.t_zer[0]
            sniffs[i_sniff]['exh_len'] = sniff_struct.t_zer[2] - sniff_struct.t_zer_fit[1]
            i_sniff += 1

    else:
        sniffs = None

    if as_dict:
        rec_id = ""
        for i in os.path.split(mat_file_path)[-1].split('_')[0:3]:
            rec_id += str(i) + "_"
        rec_id = rec_id[:-1]
        return {rec_id: sniffs}
    else:
        return sniffs


# load the spikes of a unit (for a given rec)
def load_spikes(mat_file_path, as_dict=True):
    # print (mat_file_path)
    assert (os.path.isfile(mat_file_path))
    spike_data = sio.loadmat(mat_file_path, struct_as_record=False, squeeze_me=True)
    spikes_loaded = spike_data['thisUnit']
    return np.array(spikes_loaded.times.round(), dtype=np.int)


# load trials file
def load_trials(mat_file_path, as_dict=True):
    assert (os.path.isfile(mat_file_path))
    # print (mat_file_path)
    trial_data = sio.loadmat(mat_file_path, struct_as_record=False, squeeze_me=True)
    # get the rec_id, mouse, rec, sess from the file name
    rec_id = os.path.split(mat_file_path)[-1].split('trial.mat')[0][:-1]

    trials = {}

    for x in trial_data['trial']:
        # trial     = {'rec_id'     : rec_id,
        #               'mouse'      : rec_id.split('_')[0],
        #               'rec'        : rec_id.split('_')[2],
        #               'sess'       : int(float(rec_id.split('_')[1])),
        #               'start'      : np.array([x.start for x in trial_data['trial']], dtype=np.int),
        #               'odor'       : [str(x.odorName) for x in trial_data['trial']],
        #               'odor_c'     : np.array([x.odorConc for x in trial_data['trial']], dtype=np.float),
        #               'odor_t'     : np.array([x.odorTimes for x in trial_data['trial']], dtype=np.int),
        #               'sniff_flow' : np.array([x.flow for x in trial_data['trial']], dtype=np.float),
        #               'sniff_zero' : ([x.spZeros for x in trial_data['trial']])
        #               }

        trial = {'rec_id': rec_id,
                 'mouse': rec_id.split('_')[0],
                 'rec': rec_id.split('_')[2],
                 'sess': int(float(rec_id.split('_')[1])),
                 'start': x.start,
                 'odor': str(x.odorName),
                 'odor_c': x.odorConc,
                 'odor_t': np.array(x.odorTimes, dtype=np.int),
                 'sniff_flow': np.array(x.flow, dtype=np.float),
                 'sniff_zero': np.array(x.spZeros, dtype=np.int)
                 }

        trials.update({str(x.id): trial})
    if as_dict:
        return {rec_id: trials}
    else:
        return trials


# get a record from a matlab struct of a cell
def get_rec(rec):
    cell_data = rec.cell
    cell_odor_resp = {'odors': [str(t) for t in rec.odors],
                      'trialId': [str(t) for t in rec.trialId],
                      'concs': np.array(rec.concs, dtype=np.float),
                      'spikes': np.array(rec.spikes, dtype=np.float),
                      't_0': np.array(rec.t0, dtype=np.int),
                      't_1': rec.t1,
                      't_2': rec.t2
                      }
    # print cell_meta
    cell_meta = {'light': cell_data.light,
                 'odor': cell_data.odor,
                 'quality': cell_data.quality,
                 'sessCell': cell_data.sessCell,
                 'mouse': str(cell_data.mouse),
                 'sess': cell_data.sess,
                 'rec': str(cell_data.rec),
                 'u_id': str(cell_data.uId),
                 'id': str(cell_data.Id),
                 'comment': str(cell_data.comment)
                 }

    record = {'meta': cell_meta,
              'odor_resp': cell_odor_resp,
              'light_resp': '',
              'rec_id': cell_meta['mouse'] + '_' + str(cell_meta['sess']).zfill(3) + '_' + cell_meta['rec']
              }
    return record


# Load a record (cell file)
def load_cell(mat_file_path, as_dict=False):
    assert (os.path.isfile(mat_file_path))
    cell_data = sio.loadmat(mat_file_path, struct_as_record=False, squeeze_me=True)
    exp_data_path = os.path.split(mat_file_path)[0]

    records = []
    # print type(cell_data['raster'])
    # print len(cell_data['raster'])
    # num_recs = len(cell_data['raster']) #num of recs the cell spans

    if type(cell_data['raster']) == np.ndarray:
        for rec in cell_data['raster']:
            # print rec
            record = get_rec(rec)
            records.append(record)
    else:
        records.append(get_rec(cell_data['raster']))

    for record in records:
        spikes_file_path = os.path.join(exp_data_path, record['meta']['id'] + '_spikes.mat')
        record.update({'all_spikes': load_spikes(spikes_file_path)})

    if as_dict:
        records_dict = {}
        [records_dict.update({x['meta']['id']: x}) for x in records[:]]
        return records_dict
    else:
        return records


# Load a unit file
def load_unit(mat_file_path, as_dict=True):
    assert (os.path.isfile(mat_file_path))
    unit_data = sio.loadmat(mat_file_path, struct_as_record=False, squeeze_me=True)
    unit_struct = unit_data['unit']

    # get the rec_id from the file_name
    rec_id = os.path.split(mat_file_path)[-1].split('spikes.mat')[0][:-1]

    unit = {'rec_id': rec_id,
            'mouse': rec_id.split('_')[0],
            'rec': rec_id.split('_')[2],
            'sess': int(float(rec_id.split('_')[1])),
            'u_id': str(unit_struct.uId),
            'chans': np.array(unit_struct.chans, dtype=np.int),
            'times': np.array(unit_struct.times, dtype=np.float),
            'pk_stamps': np.array(unit_struct.pkStamps, dtype=np.float),
            'clu': unit_struct.clu,
            'sites': [str(x) for x in unit_struct.sites],
            'sess_cell': unit_struct.sessCell,
            'id': rec_id + '_' + str(unit_struct.sessCell).zfill(3)
            }
    if as_dict:
        return {unit['id']: unit}
    else:
        return unit


# list the cells in a path
def list_cells(cells_path):
    all_cells = [f for f in os.listdir(cells_path) if os.path.isfile(os.path.join(cells_path, f))]
    all_units = [f[0:-9] for f in all_cells if f.find('cell.mat') > 0]
    return all_units


# get all the units
# get all units that have records, and get the corresponding trials, baselines and baseline sniffs.
def load_cells(cells_path='', cells_list=None):
    """
    :param cells_path: folder with the exportable matlab files. default is taken from fn.fold_exp_data
    :return: records (list of all the records of all the cells)
    """

    if cells_list is None:
        all_cells = [f for f in os.listdir(cells_path) if os.path.isfile(os.path.join(cells_path, f))]
        unit_files = [f for f in all_cells if f.find('cell.mat') > 0]
    else:
        unit_files = [f + '_cell.mat' for f in cells_list]

    responses = {}  # the rasters and stim sets of every unit recorded

    # rec related dictionaries
    rec_trials = {}  # the trial structures of every rec instance
    baselines = {}  # the baselines of every cell; keys of dict are record[i]['meta']['id']
    base_sniff = {}  # the sniff baselines
    # dictionary of rec related data to load
    # 'key' : [dict of the loaded data, 'tail of the filenames', loading function]
    rec_data = {'rec_trials': [rec_trials, '_trial.mat', load_trials],
                'base_sniff': [base_sniff, '_noStimSniff.mat', load_sniffs],
                }

    # unit related dictionaries
    unit_spikes = {}  # the trial structures of every rec instance
    # dictionary of unit related data to load
    unit_data = {'units': [unit_spikes, '_spikes.mat', load_unit]}

    i_f = 0
    for unit_file in unit_files:
        i_f += 1
        rec_file = os.path.join(cells_path, unit_file)
        print rec_file
        unit_recs = load_cell(rec_file, as_dict=True)
        responses.update(unit_recs)
        # get the baselines for those recs
        base_path = os.path.join(cells_path, unit_recs.itervalues().next()['meta']['u_id'] + '_spikesBase.mat')
        # print base_path
        rec_bases = load_baseline(base_path)
        # print rec_bases.keys()
        baselines.update(rec_bases)

        for a_key in unit_data.keys():
            load_function = unit_data[a_key][2]
            load_name_tail = unit_data[a_key][1]
            load_dict = unit_data[a_key][0]

            paths = [os.path.join(cells_path, unit_recs[a_rec]['meta']['id'] + load_name_tail) \
                     for a_rec in unit_recs]
            # [load_dict.update(load_function(a_path)) for a_path in paths]
        # the recording belongs to a cell (cellId)
        # TODO: get the cell (for cell signature; i.e: auto_correlogram)

        # for rec in unit_recs:
        # that recording has trials and sniffs associated.
        # get them, if they have not already been gotten
        for a_key in rec_data.keys():
            load_function = rec_data[a_key][2]
            load_name_tail = rec_data[a_key][1]
            load_dict = rec_data[a_key][0]

            paths = [os.path.join(cells_path, a_rec['rec_id'] + load_name_tail) \
                     for a_rec in unit_recs.itervalues() if a_rec['rec_id'] not in load_dict]
            # print paths
            [load_dict.update(load_function(a_path)) for a_path in paths]

    records = {'responses': responses,
               'baselines': baselines,
               'trials': rec_trials,
               'base_sniff': base_sniff}
    return records


## get a dict with the cells for one particular odor and set of concentrations
def cells_for_odor(responses, odor_aliases, odor_conc=''):
    """
   :param responses:    dictionary of responses (with unit id as keys)
           odor_aliases: alias or list of aliases to match with odor (odor name)
           odor_conc   : concentration or list of concentrations the tolerance of the match
                         is the one that is defined in conc_compare.
    :return: odor_responses dict. of all the responses found with that odor

    """
    # finds all the cells that respond to an odor, and set of concentrations
    # returns sub set of responses
    odor_responses = {}
    for key, response in responses.iteritems():
        is_right_odor = any([x in response['odor_resp']['odors'] for x in odor_aliases])
        if is_right_odor:
            if odor_conc == '':
                is_right_conc = True
            else:
                if type(odor_conc) is float:
                    odor_conc = [odor_conc]
                is_right_conc = sum([conc_compare(x, y) for x in response['odor_resp']['concs'] for y in odor_conc])

            if is_right_conc > 10:
                odor_responses.update({key: responses[key]})
    return odor_responses


## get a dict with the cells for one particular odor and set of concentrations
def cells_for_laser(responses, pow=0, dur=0):
    """
    :param responses:    dictionary of responses (with unit id as keys)
           pow (int mW):
           dur (int ms):
    :return: laser_responses dict. of all the responses found with that odor
    """
    # for now just bypasses
    return responses


# filter responses by the value of a single meta tag
def cells_by_single_tag(responses, tag, value):
    """
    :param responses:    dictionary of responses (with unit id as keys)
           odor_aliases: alias or list of aliases to match with odor (odor name)
           odor_conc   : concentration or list of concentrations the tolerance of the match
                         is the one that is defined in conc_compare.
    :return: records (list of all the records of all the cells)
    """
    filtered = {}
    [filtered.update({key: response}) for key, response in responses.iteritems() if response['meta'][tag] == value]

    return filtered


# filter responses by the value of a series of meta tags
def cells_by_tag(responses, tags):
    # filter response by a series of 'tag'=value
    if tags is not None:
        filtered = responses
        for key, value in tags.iteritems():
            filtered = cells_by_single_tag(filtered, key, value)
        return filtered
    else:
        return responses


# filter responses by the value of a single meta tag
def conc_compare(conc1, conc2, tolerance=0.45):
    return float(conc2) > (1. - tolerance) * float(conc1) and float(conc2) < (1. + tolerance) * float(conc1)


# merge responses of a rec into a single cell set
def merge_responses(responses):
    new_responses = {}
    cells_set = set([value.rec['meta']['u_id'] for value in responses.itervalues()])
    # for every cell in the set find and merge all the responsive records
    for u_id in cells_set:
        # get the list of cells with that u_id
        this_set = {u_id: [r for r in responses.itervalues() if r.rec['meta']['u_id'] == u_id]}
        new_responses.update(this_set)
    return new_responses


# some tools for handling pieces of data
# get warping parameters for a sniff loaded record
def get_warping_parameters(sniff, means=False):
    if means:
        inh_len = int(round(np.mean(sniff['inh_len'])))
        exh_len = int(round(np.mean(sniff['exh_len'])))
    else:
        inh_len = np.max(sniff['inh_len'])
        exh_len = np.max(sniff['exh_len'])
    return inh_len, exh_len


# resize a vector with interpolation (for rescaling)
def resize_chunk(chunk, new_size):
    old_size = chunk.shape[0]
    x = np.linspace(0, 1, new_size)
    xp = np.linspace(0, 1, old_size)
    y = np.interp(x, xp, chunk)
    return y
