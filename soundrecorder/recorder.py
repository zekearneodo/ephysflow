import pyaudio
import wave
import numpy as np

def unpack_bits(stream, dtype='<h', n_chans=1):
    formatted = np.fromstring(stream, dtype=np.dtype(dtype))
    return formatted

def rms(x):
    return np.linalg.norm(x)

def mad(x):
    dev = np.abs(x - np.median(x))
    return np.median(dev)

class audioBuffer:

    def __init__(self, dtype='<h', n_chans=1):
        self._stream = ''
        self.dtype = '<h'
        self.n_chans = n_chans

    def append(self, new_stream):
        self._stream = self._stream + new_stream

    def write(self, new_stream):
        self._stream = new_stream

    def read_binary(self):
        return self._stream

    # size in bytes
    def get_size(self):
        return len(self._stream)

    def get_n_formatted(self):
        return len(self._stream) / np.zeros(1, dtype=np.dtype(dtype)).nbytes

    def clear_data(self):
        self._stream = ''

    def read_formatted(self):
        return unpack_bits(self._stream, self.dtype, self.n_chans)


class Recorder:


    def __init__(self, channels=1, rate=44100, frames_per_buffer=1024):
        self.channels = channels
        self.rate = rate
        self.sampling_step_ms = 1000./rate
        self.frames_per_buffer = frames_per_buffer
        self._pa = pyaudio.PyAudio()
        self._stream = None
        self.stream_buffer = audioBuffer(dtype='<h', n_chans = channels)

        self._is_recording = False
        self.rms_thresh = 5000
        self.rms_stop_thresh = 1000
        self.monitor_channel = 0
        self.monitor_buffer_size_ms = 1500
        self.monitor_buffer_max_elem = self.monitor_buffer_size_ms/self.sampling_step_ms * channels
        self.monitor_status = 'idle'
        self.recorded_ms = 0
        self.record_epoch_ms = 60000 #records maximum 60 sec epochs
        self.ms_in_buf = 0

    def msec_to_frames(self, n_msec):
        return np.int(np.ceil(self.rate/(1000.*self.frames_per_buffer)*n_msec))


    def read_frames(self, n, msec=True):
        self._stream = self._pa.open(format=pyaudio.paInt16,
                                    channels=self.channels,
                                    rate=self.rate,
                                    input=True,
                                    frames_per_buffer=self.frames_per_buffer)
        self._is_recording = True
        n_frames = self.msec_to_frames(n) if msec else n_frames
        self.stream_buffer.clear_data()
        for frame in range(n_frames):
            self.stream_buffer.append(self._stream.read(self.frames_per_buffer))
        self._is_recording = False
        return self.stream_buffer

    def get_avg_rms(self, window_len = 3000):
        rms_buffer_formatted = self.read_frames(3000).read_formatted()

        return rms(rms_buffer_formatted), np.median(rms_buffer_formatted), mad(rms_buffer_formatted)


    def start_triggered_monitor(self):
        print("starting recording")
        self._stream = self._pa.open(format=pyaudio.paInt16,
                            channels=self.channels,
                            rate=self.rate,
                            input=True,
                            frames_per_buffer=self.frames_per_buffer,
                            stream_callback=self.get_callback())


        self.stream_buffer.clear_data()
        self.monitor_status = 'idle'
        #try:

        self._stream.start_stream()
        #except KeyboardInterrupt:
            # The recording was interrupted by ctrl+c
            #self._stream.stop_stream()
              #self.stop_triggered_recording()
        return self

    def get_callback(self):
        def callback(in_data, frame_count, time_info, status):
            # load in the buffer and check if it has to do something
            print ("Monitoring")
            self.audioBuffer.append(in_data)

            samples_in_buf = self.audioBuffer.get_n_formatted()/self.channels
            self.ms_in_buf = samples_in_buf * self.sampling_step_ms

            # Check if it has to decide on the buffer (i.e, it went beyoind buffer_max_elem)
            if self.ms_in_buf > self.monitor_buffer_size_ms:
                data_formatted = unpack_bits(in_data, dtype=self.dtype, n_chans=self.channels)
                plt.plot(data_formatted)
                rms_buf = rms(data_formatted)

                if self.monitor_status == 'idle':
                    if rms_buf > self.rms_thresh:
                        # has to begin recording
                        self.start_triggered_recording()

                elif self.monitor_status == 'recording':
                    self.buffer_to_file()
                    self.recorded_ms = self.recorded_ms + self.ms_in_buf
                    # check if it has to stop recording
                    if rms_buf < self.rms_stop_thresh | self.recorded_ms > self.record_epoch_ms:
                        self.stop_triggered_recording()

                # Outway of the state machine, has to reset the buffer
                self.audioBuffer.clear_data()
            # end of buffer_triggered state machine
            return in_data, pyaudio.paContinue
        return callback

    def start_triggered_recording(self):
        print("Recording Started")
        self.prep_file()
        self.buffer_to_file()
        self.recorded_ms = self.ms_in_buf

    def stop_triggered_recording(self):
        print("Recording Stopped")
        self.buffer_to_file()
        self.close_file()
        self.monitor_status = 'idle'

    def make_file_name(self):
        return 'tst.wav'

    def prep_file(self):
        file_name = self.make_file_name()
        print('preparing file' + file_name)

    def buffer_to_file(self):
        # send the data to file
        print("storing data")
        #plt.plot(self.stream_buffer.read_formatted())
        self.stream_buffer.clear_data()