import os
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy.video.io.VideoFileClip import VideoFileClip

# import subprocess

# try:
#     subprocess.run(['ffmpeg', '-version'], check=True)
#     print("FFmpeg is installed and accessible.")
# except FileNotFoundError:
#     print("FFmpeg is not installed or cannot be found.")
# except subprocess.CalledProcessError:
#     print("FFmpeg returned a non-zero exit status.")

start_time = 0 * 60
end_time = 5 * 60

input_video = r"C:\Users\Heng2020\OneDrive\Python NLP\InputData\BigBang PT S03E01.mkv"
folder_path = r"C:\Users\Heng2020\OneDrive\Python NLP\OutputData"

output_video = 'S03E01_short01.mp4'

output_path = os.path.join(folder_path,output_video)

# ffmpeg_extract_subclip(input_video, start_time, end_time, targetname=output_video)

with VideoFileClip(input_video) as video:
    new = video.subclip(start_time, end_time)
    new.write_videofile(output_path, audio_codec='aac')
