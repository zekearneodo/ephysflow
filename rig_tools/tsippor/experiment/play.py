"""PyAudio Example: Play a wave file (callback version)"""

import pyaudio
import wave
import time
import sys
import Adafruit_BBIO.GPIO as GPIO



if len(sys.argv) < 2:
    print("Plays a wave file.\n\nUsage: %s filename.wav" % sys.argv[0])
    sys.exit(-1)

wf = wave.open(sys.argv[1], 'rb')

p = pyaudio.PyAudio()

first_chunk = True
GPIO.setup("P8_10", GPIO.OUT)

def callback(in_data, frame_count, time_info, status):
    global first_chunk
    if first_chunk:
        GPIO.output("P8_10", GPIO.HIGH)
	first_chunk = False

    data = wf.readframes(frame_count)
    return (data, pyaudio.paContinue)

stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                channels=wf.getnchannels(),
                rate=wf.getframerate(),
                output=True,
                stream_callback=callback)

GPIO.output("P8_10", GPIO.HIGH)
stream.start_stream()

while stream.is_active():
    time.sleep(0.1)

stream.stop_stream()
stream.close()
print "stream closed"
GPIO.setup("P8_10", GPIO.OUT)
wf.close()

p.terminate()
GPIO.output("P8_10", GPIO.LOW)
#GPIO.cleanup()
