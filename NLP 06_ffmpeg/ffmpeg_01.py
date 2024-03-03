# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 10:20:59 2023

@author: Heng2020
"""

from pathlib import Path
from langcodes import Language
import langcodes
import pycountry
from typing import Union,List,Tuple, Callable
import subprocess
import sys
import logging

# import string_01 as st

# NEXT: extract the Ark subtitle
# The matrix subtitle(muti langauge in 1 video)
sys.path.append(r'C:\Users\Heng2020\OneDrive\Python MyLib\Python MyLib 01\10 OS')

import os_01 as ost

from typing import Literal, Union
def extract_audio2(
        video_folder:     Union[Path,str],
        output_folder:    Union[Path,str],
        video_extension:  Union[list,str] = [".mp4",".mkv"],
        output_extension: Union[list,str] = ".mp3",
        overwrite_file:   bool = True,
        n_limit:          int = 150,
        output_prefix:    str = "",
        output_suffix:    str = "",
        play_alarm:       bool = True,
        alarm_done_path:str = r"H:\D_Music\Sound Effect positive-logo-opener.mp3"
):
    input_param = {
        'video_path': 6
    }

    _extract_media_setup(
        input_folder = video_folder,
        output_folder = output_folder,
        input_extension = video_extension,
        output_extension = output_extension,
        extract_1_file_func = extract_1_audio,
        overwrite_file = overwrite_file,
        n_limit = n_limit,
        output_prefix = output_prefix,
        output_suffix = output_suffix,
        play_alarm = play_alarm,
        alarm_done_path = alarm_done_path,
    )

def extract_subtitle(
        video_folder:     Union[Path,str],
        output_folder:    Union[Path,str],
        video_extension:  Union[list,str] = [".mp4",".mkv"],
        output_extension: Union[list,str] = None,
        overwrite_file:   bool = True,
        n_limit:          int = 150,
        output_prefix:    str = "",
        output_suffix:    str = "",
        play_alarm:       bool = True,
        alarm_done_path:str = r"H:\D_Music\Sound Effect positive-logo-opener.mp3"
):
    input_param = {
        'video_path': 6
    }
    
    _extract_media_setup(
        input_folder = video_folder,
        output_folder = output_folder,
        input_extension = video_extension,
        output_extension = output_extension,
        extract_1_file_func = extract_sub_1_video,
        overwrite_file = overwrite_file,
        n_limit = n_limit,
        output_prefix = output_prefix,
        output_suffix = output_suffix,
        play_alarm = play_alarm,
        alarm_done_path = alarm_done_path,
    )

def _extract_media_setup(
        input_folder: Union[str,Path],
        output_folder: Union[str,Path],
        extract_1_file_func: Callable,
        input_extension: Union[list[str],str],
        output_extension: Union[list[str],str],
        # input_param_name: list[str],
        overwrite_file:   bool = True,
        n_limit: int = 150,
        output_prefix:    str = "",
        output_suffix:    str = "",
        play_alarm: bool = True,
        alarm_done_path:str = r"H:\D_Music\Sound Effect positive-logo-opener.mp3"
):
    """
    helper function to reduce code redundancy
    it would setup which/ how many files should be extracted in inputs
    how many files should be created in output 


    """
    import sys
    from pathlib import Path
    from playsound import playsound
    from time import time, perf_counter
    sys.path.append(r'C:\Users\Heng2020\OneDrive\Python MyLib\Python MyLib 01\08 Other')
    import lib08_Other as pw

    ts01 = time()
    output_extension = [output_extension]
    output_extension_in = []
    
    # add . to extension in case it doesn't have .
    if output_extension[0] is not None:
        for extension in output_extension:
            if not "." in extension:
                output_extension_in.append("."+extension)
            else:
                output_extension_in.append(extension)
    else:
        output_extension_in = [None]


    filename_list_ext = ost.get_filename(input_folder,input_extension)
    path_list = ost.get_full_filename(input_folder,input_extension)
    # warrus operator, makes it usuable only for python >= 3.8
    (n_file := min(len(filename_list_ext),n_limit))
    filename_list_ext = filename_list_ext[:n_file]
    path_list = path_list[:n_file]

    filename_list = [filename.split('.')[0] for filename in filename_list_ext]

    for i, filename in enumerate(filename_list):
        
            
        output_name = output_prefix + filename_list[i] + output_suffix
        # original_stdout = sys.stdout
        # sys.stdout = open('nul', 'w')
         
        # the problem here is that the input parameter name in extract_1_file_func
        # could be different and 

        for j, extension in enumerate(output_extension_in):
            # input_dict = {
            #     input_param_name[0]:path_list[i],
            #     input_param_name[1]:extension,
            # }
            extract_1_file_func(
                video_path = path_list[i],
                output_extension = extension,
                output_folder = output_folder,
                output_name = output_name,
                play_alarm=False,
                overwrite_file=overwrite_file)
            print(f"extracted {output_name} successfully!!!")
        
        # sys.stdout = original_stdout
    if play_alarm:
        playsound(alarm_done_path)
    ts02 = time()
    duration = ts02-ts01
    pw.print_time(duration)
    print()
    return filename_list

def extract_sub_1_video(
    video_path:         Union[str,Path],
    output_folder:      Union[str,Path],
    output_name:        Union[str,Path] = None, 
    output_extension:   Union[str,list] = None,
    play_alarm:         bool = True,
    overwrite_file:     bool = True,
    language:           Union[str,list, None] = None,
    alarm_done_path:    str = r"H:\D_Music\Sound Effect positive-logo-opener.mp3"
                    ):
    # medium tested
    # ToAdd feature 01: extract mutiple subtitles for many languages
    # ToAdd feature 02: select only some languages to extract
    
    
    """
    Extract audio from a video file and save it in the specified format.
    
    Parameters:
    -----------
    video_path : str or Path
        The path to the input video file.
        
    output_folder : str or Path
        The folder where the extracted audio file will be saved.
        
    output_name : str
        The name of the output audio file (without extension).
        
    file_extension : str, optional
        The desired file extension for the output audio file (default is ".mp3").
        
    play_alarm : bool, optional
        Whether to play an alarm sound upon successful extraction (default is True).
        
    overwrite_file : bool, optional
        Whether to overwrite the output file if it already exists (default is True).
    
    Returns:
    --------
    bool
        True if audio extraction is successful, False otherwise.
    
    Notes:
    ------
    - Additional feature 1: Output both .wav & .mp3 formats.
    - This function relies on FFmpeg for audio extraction, so make sure FFmpeg is installed.
    - The codec for output format is determined based on the file_extension parameter.
    - An alarm sound is played if play_alarm is set to True upon successful extraction.
    - If the output file already exists and overwrite_file is set to False, the function will return False.
    
    Example:
    --------
    extract_1_audio("input_video.mp4", "output_folder", "output_audio", file_extension=".wav")
    
    """
    
    from pathlib import Path
    import subprocess
    from playsound import playsound
    import os
    # only input language as str for now
    
    output_folder_in = Path(output_folder)

    video_name = ost.extract_filename(video_path,with_extension=False)
    ori_extension = get_subtitle_extension(video_path,language)

    if output_extension is None:
        if output_name is None:
            output_name = video_name
        if ori_extension not in output_name:
            if "." not in ori_extension:
                ori_extension = "." + ori_extension
            output_name += ori_extension


    elif isinstance(output_extension, str):

        if output_name is None:
            output_name = video_name

        if output_extension not in output_name:
            
            if "." not in output_extension:
                output_extension = "." + output_extension
            output_name += output_extension
    
    output_path = output_folder_in / output_name
    
    subtitle_stream_index = get_subtitle_index(video_path,language)
    # from extract_1_audio
    # command = [
    #     "ffmpeg",
    #     "-i", str(video_path),
    #     # "-map", "0:a:m:language:por",
    #     "-c:a", codec,
    #     "-q:a", "0",
    #     str(output_path)
    # ]
    if output_extension:
        output_ext_no_dot = output_extension.replace('.','')
    else:
        output_ext_no_dot = ori_extension.replace('.','')
    command = [
        'ffmpeg',
        '-i', str(video_path),  # Input file
        '-map', f'0:{subtitle_stream_index}',  # Map the identified subtitle stream
        '-c:s', output_ext_no_dot,  # Subtitle format
        str(output_path)
    ]
    # cmd_line is for debugging
    cmd_line = ' '.join(command)
    
    if os.path.exists(str(output_path)):
        if overwrite_file:
            os.remove(str(output_path))
        else:
            print("The output path is already existed. Please delete the file or set the overwrite parameter to TRUE")
            return False
    result = subprocess.run(command, text=True, stderr=subprocess.PIPE)
    
    if result.returncode != 0:
        print("Error encountered:")
        print(result.stderr)
    
    elif result.returncode == 0:
        # print("Extract audio successfully!!!")
        
        if play_alarm:
            playsound(alarm_done_path)

def crop_video(
        video_path: str, 
        t_start: str, 
        t_end: str, 
        time_slice: List[Tuple[str, str]],
        output_extension: Literal["mp3", ".mp3",".mp4","mp4","mkv",".mkv","wav",".wav"] = None,
        play_alarm = True
        ):
    # tested only input(mkv) => output(mkv)
    import subprocess
    import os
    from playsound import playsound
    
    
    alarm_done_path = r"H:\D_Music\Sound Effect positive-logo-opener.mp3"
    
    # Construct the base output filename
    base_name = os.path.splitext(video_path)[0]
    
    if output_extension is None:
        extension_in = os.path.splitext(video_path)[1][1:]
    else:
        extension_in = (output_extension.split(".")[1]) if "." in output_extension else output_extension
    # Find an unused file name
    i = 1
    while os.path.exists(f"{base_name}_{i:02d}.{extension_in}"):
        i += 1
    output_path = f"{base_name}_{i:02d}.{extension_in}"
    # FFmpeg command
    command = [
        'ffmpeg', '-ss', t_start, '-to', t_end,
        '-i', video_path,
        '-c', 'copy' if extension_in in ["mp4","mkv"] else '-vn',
        output_path
    ]
    
    result = subprocess.run(command, text=True, stderr=subprocess.PIPE)
    
    if result.returncode != 0:
        print("Error encountered:")
        print(result.stderr)
    
    elif result.returncode == 0:
        print("Extract audio successfully!!!")
        
        if play_alarm:
            playsound(alarm_done_path)
    
    return output_path  # Return the output file path



def is_ffmpeg_installed():
    
    import subprocess
    try:
        # Run the 'ffmpeg -version' command
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, check=True)
        # If the above command runs successfully, FFmpeg is installed and in PATH
        print("FFmpeg is installed and accessible in PATH.")
    except subprocess.CalledProcessError:
        # An error occurred while running FFmpeg, it might not be installed or in PATH
        print("FFmpeg is not installed.")
    except FileNotFoundError:
        # FFmpeg is not in PATH
        print("FFmpeg is installed but not in PATH.")

def get_metadata3(file_path):
    # this is older version use get_metadata
    try:
        # Constructing the FFmpeg command
        command = [
            "ffmpeg",
            "-i", str(file_path),
            "-hide_banner"
        ]

        # Running the command and capturing the output
        result = subprocess.run(command, text=True, stderr=subprocess.PIPE, stdout=subprocess.DEVNULL)

        # The metadata and other info are printed to stderr
        return result.stderr
    except subprocess.CalledProcessError as e:
        # Handle errors (such as file not found, etc.)
        print("An error occurred:", e)
        return None



def closest_language(misspelled_language):
    
    from fuzzywuzzy import process
    import pycountry
    # Get a list of all language names
    language_names = [lang.name for lang in pycountry.languages if hasattr(lang, 'name')]

    # Use fuzzy matching to find the closest match
    closest_match = process.extractOne(misspelled_language, language_names)
    return closest_match[0] if closest_match else None

def closest_language_obj(misspelled_language):
    
    """
    Find the closest matching language object for a potentially misspelled language code.
    
    Parameters:
    -----------
    misspelled_language : str
        The potentially misspelled language code.
    
    Returns:
    --------
    langcodes.Language
        A language object representing the closest matching language.
    
    Notes:
    ------
    - This function uses the 'langcodes' library to find the closest matching language object
      for a potentially misspelled language code.
    - It can be useful for language code correction or normalization.
    
    Example:
    --------
    >>> closest_language_obj("englsh")
    <Language('en', 'English')>
    >>> closest_language_obj("español")
    <Language('es', 'Spanish')>
    
    """
    
    
    from langcodes import Language
    correct_language = closest_language(misspelled_language)
    return Language.find(correct_language)
    
def extract_1_audio(video_path:     Union[str,Path],
                    output_folder:  Union[str,Path],
                    output_name:    Union[str,Path], 
                    output_extension: Union[str,list] = ".mp3",
                    play_alarm:     bool = True,
                    overwrite_file: bool = True,
                    alarm_done_path = r"H:\D_Music\Sound Effect positive-logo-opener.mp3"
                    ):
    # ToAdd feature 01: output both .wav & .mp3
    
    
    """
    Extract audio from a video file and save it in the specified format.
    
    Parameters:
    -----------
    video_path : str or Path
        The path to the input video file.
        
    output_folder : str or Path
        The folder where the extracted audio file will be saved.
        
    output_name : str
        The name of the output audio file (without extension).
        
    file_extension : str, optional
        The desired file extension for the output audio file (default is ".mp3").
        
    play_alarm : bool, optional
        Whether to play an alarm sound upon successful extraction (default is True).
        
    overwrite_file : bool, optional
        Whether to overwrite the output file if it already exists (default is True).
    
    Returns:
    --------
    bool
        True if audio extraction is successful, False otherwise.
    
    Notes:
    ------
    - Additional feature 1: Output both .wav & .mp3 formats.
    - This function relies on FFmpeg for audio extraction, so make sure FFmpeg is installed.
    - The codec for output format is determined based on the file_extension parameter.
    - An alarm sound is played if play_alarm is set to True upon successful extraction.
    - If the output file already exists and overwrite_file is set to False, the function will return False.
    
    Example:
    --------
    extract_1_audio("input_video.mp4", "output_folder", "output_audio", file_extension=".wav")
    
    """
    
    from pathlib import Path
    import subprocess
    from playsound import playsound
    import os
    
    codec_dict = {'.mp3': "libmp3lame",
                  'mp3' : "libmp3lame",
                  '.wav': "pcm_s24le",
                  'wav' : "pcm_s24le"
                  }
    
    codec = codec_dict[output_extension]
    
    output_folder_in = Path(output_folder)
    
    if isinstance(output_extension, str):
        if output_extension not in output_name:
            
            if "." not in output_extension:
                output_extension = "." + output_extension
            output_name += output_extension
    
    output_path = output_folder / output_name
    
    
    
    command = [
        "ffmpeg",
        "-i", str(video_path),
        # "-map", "0:a:m:language:por",
        "-c:a", codec,
        "-q:a", "0",
        str(output_path)
    ]
    
    if os.path.exists(str(output_path)):
        if overwrite_file:
            os.remove(str(output_path))
        else:
            print("The output path is already existed. Please delete the file or set the overwrite parameter to TRUE")
            return False
    result = subprocess.run(command, text=True, stderr=subprocess.PIPE)
    
    if result.returncode != 0:
        print("Error encountered:")
        print(result.stderr)
    
    elif result.returncode == 0:
        print("Extract audio successfully!!!")
        
        if play_alarm:
            playsound(alarm_done_path)


def extract_audio(video_folder:     Union[Path,str],
                  output_folder:    Union[Path,str],
                  video_extension:  Union[list,str] = [".mp4",".mkv"],
                  output_extension: Union[list,str] = ".mp3",
                  output_prefix:    str = "",
                  output_suffix:    str = "",
                  play_alarm:       bool = True,
                  overwrite_file:   bool = True,
                  n_limit:          int = 150
                  ):
    # TODO 
    # add feature: support multiple languages
    # support multiple output eg [.wav,.mp3,.eac3]
    
    """
    Extracts audio from video files in the specified `video_folder` and saves them in the `output_folder` in the specified audio format.
    
    Parameters
    ----------
    video_folder : Union[Path, str]
        The path to the folder containing video files.
        
    output_folder : Union[Path, str]
        The path where extracted audio files will be saved.
        
    video_extension : Union[list, str], optional
        List of video file extensions to consider for extraction. Defaults to [".mp4", ".mkv"].
        
    output_extension : Union[list, str], optional
        The audio file extension for the output files. Defaults to ".mp3".
        
    output_prefix : str, optional
        A prefix to be added to the output audio file names. Defaults to an empty string.
        
    output_suffix : str, optional
        A suffix to be added to the output audio file names. Defaults to an empty string.
        
    play_alarm : bool, optional
        Whether to play an alarm sound when the extraction is completed. Defaults to True.
        
    overwrite_file : bool, optional
        Whether to overwrite existing audio files with the same name in the `output_folder`. Defaults to True.
        
    n_limit : int, optional
        The maximum number of video files to process. Defaults to 150.
        
    Returns
    -------
    """
    
    import sys
    from pathlib import Path
    from playsound import playsound
    from time import time
    ts01 = time()
    
    alarm_done_path = r"H:\D_Music\Sound Effect positive-logo-opener.mp3"
    sys.path.append(r'C:\Users\Heng2020\OneDrive\Python MyLib\Python MyLib 01\08 Other')
    import lib08_Other as pw
    
    codec_dict = {'.mp3': "libmp3lame",
                  'mp3' : "libmp3lame",
                  '.wav': "pcm_s24le",
                  'wav' : "pcm_s24le"
                  }
    
    output_extension = [output_extension]
    output_extension_in = []
    
    # add . to extension in case it doesn't have .
    for extension in output_extension:
        if not "." in extension:
            output_extension_in.append("."+extension)
        else:
            output_extension_in.append(extension)
    
    video_name_list_ext = ost.get_filename(video_folder,video_extension)
    video_path_list = ost.get_full_filename(video_folder,video_extension)
    
    n_file = min(len(video_name_list_ext),n_limit)
    video_name_list_ext = video_name_list_ext[:n_file]
    video_path_list = video_path_list[:n_file]
    
    video_name_list = [filename.split('.')[0] for filename in video_name_list_ext]
    
    for i, video_name in enumerate(video_name_list):
        
            
        output_name = output_prefix + video_name_list[i] + output_suffix
        # original_stdout = sys.stdout
        # sys.stdout = open('nul', 'w') 
        
        # fix i to j
        for j, extension in enumerate(output_extension_in):
            extract_1_audio(
                video_path = video_path_list[i],
                output_folder = output_folder,
                output_name = output_name,
                output_extension = extension,
                play_alarm=False,
                overwrite_file=overwrite_file)
        
        # sys.stdout = original_stdout
        
    if play_alarm:
        playsound(alarm_done_path)
    ts02 = time()
    duration = ts02-ts01
    pw.print_time(duration)
    
    return video_name_list



def get_metadata2(media_path):
    import subprocess
    import json
    # 80% from GPT4
    """
    Get the index of the first subtitle stream in the video file.
    
    Parameters:
    - video_path: Path to the input video file.
    
    Returns:
    - Index of the first subtitle stream, or None if no subtitle stream is found.
    """
    command = [
        'ffprobe',
        '-v', 'quiet',
        '-show_streams',
        media_path
    ]
    
    result = subprocess.run(command, check=True, stdout=subprocess.PIPE, text=True)
    streams_info_raw = json.loads(result.stdout)
    
    streams_info = [stream for stream  in streams_info_raw['streams']]

    
    return streams_info

def get_all_metadata(media_path):
    import subprocess
    import json    
    import pandas as pd
    #  !!!!!!!!!!!! this is the main get_metadata
    # medium tested
    # 100% from GPT4
    # new and updated version
    

    """
    Get metadata from a media file and return it as a pandas DataFrame.
    
    Parameters:
    - media_path: Path to the input media file.
    
    Returns:
    - DataFrame with columns for 'filetype', 'file_extension', 'language', and 'duration'.
    """
    command = [
        'ffprobe',
        '-v', 'quiet',
        '-print_format', 'json',
        '-show_streams',
        '-show_format',
        media_path
    ]
    
    result = subprocess.run(command, check=True, stdout=subprocess.PIPE, text=True)
    metadata = json.loads(result.stdout)
    
    # Initialize lists to hold data for each column
    filetypes = []
    file_extensions = []
    languages = []
    durations = []
    
    # Extract stream information
    for stream in metadata.get('streams', []):
        filetypes.append(stream.get('codec_type'))
        file_extensions.append(stream.get('codec_name'))
        # Extract language; note that 'tags' and 'language' might not exist
        language = stream.get('tags', {}).get('language', 'N/A')
        languages.append(language)
    
    # Extract duration from format, if available
    duration = float(metadata.get('format', {}).get('duration', 'N/A')) / 60
    durations = [duration] * len(filetypes)  # Replicate duration for all rows
    
    # Create DataFrame
    info_df = pd.DataFrame({
        'filetype': filetypes,
        'file_extension': file_extensions,
        'language': languages,
        'duration_in_min': durations
    })
    
    return info_df

def get_metadata(media_path, media, language = None, file_extension = None):
    #  not tested
    if language is None:
        language_in = None
    elif not isinstance(language, list):
        language_in = [language]
    else:
        language_in = list(language)
    
    if file_extension is None:
        file_extension_in = None
    elif not isinstance(file_extension, list):
        # remove '.' from the file_extension
        file_extension_in = file_extension.replace('.','')
        file_extension_in = [file_extension_in]
    else:
        file_extension_in = [extension.replace('.','') for extension in file_extension]
    
        
    # requires get_metadata
    media_info = get_all_metadata(media_path)
    
    if language_in:
        if file_extension_in:
            selected_media = media_info.loc[(media_info['filetype'] == media) 
                                            & media_info['language'].isin(language_in)
                                            & media_info['language'].isin(file_extension)
                                            ]
        else:
            selected_media = media_info.loc[(media_info['filetype'] == media) & media_info['language'].isin(language_in)  ]
    else:
        
        if file_extension_in:
            selected_media = media_info.loc[(media_info['filetype'] == media) 
                                            & media_info['language'].isin(file_extension)
                                            ]
        else:
            selected_media = media_info.loc[(media_info['filetype'] == media) ]
            
    return selected_media

def _get_media_extension(media_path, media, language = None, file_extension = None
                         ) -> Union[list[int],int, None] :
    # not tested
    # return the unique list of media extension
    # return str if 1 unique extension is found
    selected_media = get_metadata(media_path, media, language = language, file_extension = file_extension)
    # subrip is the same as .srt
    # so I converted to srt
    selected_media.loc[selected_media["file_extension"].isin(["subrip"]),"file_extension"] = "srt"
    unqiue_ext = list(set(selected_media['file_extension'].tolist()))
    
    if len(unqiue_ext) == 0:
        return None
    elif len(unqiue_ext) == 1:
        return unqiue_ext[0]
    else:
        return unqiue_ext

def get_video_extension(media_path, file_extension = None):
    return _get_media_extension(media_path,'video')

def get_audio_extension(media_path, language = None, file_extension = None):
    return _get_media_extension(media_path,'audio',language)

def get_subtitle_extension(media_path, language = None, file_extension = None):
    return _get_media_extension(media_path,'subtitle',language)


def _get_media_index(media_path, media, language = None, file_extension = None):
    
    selected_media = get_metadata(media_path, media, language = None, file_extension = None)
    idx_list = selected_media.index.tolist()
    # return None if media is not found
    if len(idx_list) == 0:
        return None
    elif len(idx_list) == 1:
        return idx_list[0]
    else:
        return idx_list

def get_video_index(media_path, file_extension = None):
    return _get_media_index(media_path,'video')

def get_audio_index(media_path, language = None, file_extension = None):
    return _get_media_index(media_path,'audio',language)

def get_subtitle_index(media_path, language = None, file_extension = None):
    return _get_media_index(media_path,'subtitle',language)



# ################################################################################

def test_get_metadata():
    
    folder = Path(r"H:\D_Video\The Ark Season 01 Portuguese")
    video_name = "The Ark S01E02 PT.mkv"
    video_path = folder / video_name
    test = get_all_metadata(video_path)
    logging.debug('Done From test_get_subtitle_stream_index')

def test_get_subtitle_index():
    
    folder = Path(r"H:\D_Video\The Ark Season 01 Portuguese")
    video_name = "The Ark S01E02 PT.mkv"
    video_path = folder / video_name
    actual01 = get_subtitle_index(video_path)
    
    
    folder = Path(r"E:\Videos\The Big Bang Theory\The Big Bang Theory French Season 06")
    video_name = "The Big Bang Theory French S06E01.mp4"
    video_path = folder / video_name
    actual02 = get_subtitle_index(video_path)
    
    
    
    
    logging.debug('Done From test_get_subtitle_index') 


def test_extract_1_audio():
    
    folder = Path(r"E:\Videos\The Big Bang Theory\The Big Bang Theory French Season 06")
    video_name = "The Big Bang Theory French S06E01.mp4"
    video_path = folder / video_name
    output_folder = Path(r"C:\Users\Heng2020\OneDrive\Python NLP\NLP 06_ffmpeg")
    output_name = "The Big Bang Theory French S06E01.mp3"
    output_path = output_folder / output_name
    
    extract_1_audio(video_path,output_folder,output_name)
    extract_1_audio(video_path,output_folder,output_name,overwrite_file = False)

def test_extract_audio():
    
    from pathlib import Path
    French_bigbang = Path(r"E:\Videos\The Big Bang Theory\The Big Bang Theory French Season 06")
    output_folder = Path(r"E:\Videos\The Big Bang Theory\The Big Bang Theory French Season 06\Audio")
    extract_audio(French_bigbang,output_folder,n_limit=10)
    return True

def test_crop_video():
    from pathlib import Path
    video_path = r"C:\Users\Heng2020\OneDrive\Python NLP\InputData\Westworld S04E01 Portuguese.mkv"
    t1 = "0:02:25"
    t2 = "0:06:00"
    
    crop_video(video_path,t1,t2)

def test_create_subtitle():
    import whisper
    from whisper.utils import get_writer
    from playsound import playsound
    from time import time
    
    alarm_done_path = r"H:\D_Music\Sound Effect positive-logo-opener.mp3"
    
    ts01 = time()
    
    input_path = r"C:\Users\Heng2020\OneDrive\Python NLP\InputData\Westworld S04E01 Portuguese_01.mkv"
    
    model = whisper.load_model("base")
    result = model.transcribe(input_path)
    
    output_directory = r"C:\Users\Heng2020\OneDrive\Python NLP\InputData\Westworld S04E01 Portuguese_01.srt"
    
    writer = get_writer("srt", str(output_directory))
    writer(result, output_name)
    ts02 = time()
    
    duration = ts02 - ts01 
    print(duration)




# command = [
#     "ffmpeg",
#     "-i", str(video_path),
#     "-map", "0:2",
#     "-c:a", "libmp3lame",
#     "-q:a", "0",
#     str(output_path)
# ]

# command = [
#     "ffmpeg",
#     "-i", str(video_path),
#     "-map", "0:a:m:language:por",
#     "-c:a", "copy",
#     str(output_path)
# ]

# command = [
#     "ffmpeg",
#     "-i", str(video_path),
#     # "-map", "0:a:m:language:por",
#     "-c:a", "libmp3lame",
#     "-q:a", "0",
#     str(output_path)
# ]

def main():
    logging.basicConfig(level=logging.DEBUG)
    test_get_subtitle_index()

    # test_extract_1_audio()

    # test_create_subtitle()
    raise 

    folder = Path(r"E:\Videos\The Big Bang Theory\The Big Bang Theory French Season 06")

    video_name = "The Big Bang Theory French S06E01.mp4"

    video_path = folder / video_name

    video_path02 = r"H:\D_Video\BigBang Portugues\BigBang PT Season 06\BigBang PT S06E01.mkv"
    video_path03 = r"H:\D_Video\The Matrix Resurrections 2021.mkv"

    output_folder = Path(r"C:\Users\Heng2020\OneDrive\Python NLP\NLP 06_ffmpeg")

    output_name = "The Big Bang Theory French S06E01.mp3"
    output_path = output_folder / output_name
    
    extract_1_audio(video_path,output_folder,output_name,overwrite_file = False)

    # command = ['efef']
    subprocess.run(command)

    cmd_line = ' '.join(command)
    cmd_line

    result = get_all_metadata(video_path)

    result02 = get_all_metadata(video_path02)
    result03 = get_all_metadata(video_path03)

    lang = closest_language_obj('Portugues')
    test = lang.to_alpha3()

    languages = [language.name for language in pycountry.languages if hasattr(language, 'name')]
    print("Portuguese" in languages)

    languages.index("Portuguese")

if __name__ == '__main__':
    main()
    



