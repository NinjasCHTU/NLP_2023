import ffmpeg
import os
import wave
def temp_fun(x):
    return x

from moviepy.editor import VideoFileClip
folder_path = r"C:\Users\Heng2020\OneDrive\Python NLP\OutputData"

input_full = r"C:\Users\Heng2020\OneDrive\Python NLP\InputData\BigBang PT S03E01.mkv"
output_name = 'PT01.wav'

output_path = os.path.join(folder_path,output_name)


# Load the MKV file
video = VideoFileClip(input_full)

# Get the Portuguese audio track by track index (0 for the first track)
audio_eng = video.audio.subclip(audio_tracks=[0])
audio_por = video.audio.subclip(audio_tracks=[1])

# Save the Portuguese audio track as WAV file
audio_por.write_audiofile(output_path)


input_stream = ffmpeg.input(input_full)
input_stream_audio = input_stream.audio
# Extract the Portuguese audio track using the "-map" option
output_stream = ffmpeg.output(input_stream['a:por'], output_path)

# Run the extraction process
output_stream.run(output_path)


# Open the WAV file for writing
with wave.open(output_path, 'wb') as wav_file:
    # Set the audio file parameters
    # wav_file.setnchannels(output_stream.channels)
    wav_file.setsampwidth(output_stream.sample_width)
    wav_file.setframerate(output_stream.sample_rate)

    # Write the audio data to the file
    wav_file.writeframes(output_stream.getvalue())
