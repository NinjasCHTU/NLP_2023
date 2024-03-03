import unittest
from ffmpeg_01 import *

should_skip = True

# @unittest.skipIf(should_skip, "Skipping: Test_extract_subtitle")
class Test_extract_subtitle(unittest.TestCase):
    folder01 = Path(r"H:\D_Video\The Ark Season 01 Portuguese")

    output_folder01 = Path(r'H:\D_Video\The Ark Season 01 Portuguese\Subtitles')

    def test_basic01(self):
        extract_subtitle(self.folder01, self.output_folder01)

def test_extract_subtitle():
# extract_subtitle doesn't work on just 1 video
    
    # folder01 = Path(r"H:\D_Video\The Ark Season 01 Portuguese")
    # output_folder01 = Path(r'H:\D_Video\The Ark Season 01 Portuguese\Subtitles')
    # extract_subtitle(folder01,output_folder01)

    # folder02 = Path(r"C:\Users\Heng2020\OneDrive\Python NLP\InputData\The Matrix Resurrections 2021.mkv")
    # output_folder02 = Path(r'C:\Users\Heng2020\OneDrive\Python NLP\OutputData\SubtitleMatrix')
    # extract_subtitle(folder02,output_folder02)

    # test when input is .ass but output is .srt
    folder03 = Path(r"H:\D_Video\The Ark Season 01 Portuguese")
    output_folder03 = Path(r'C:\Users\Heng2020\OneDrive\Python NLP\OutputData\The Ark season 1 srt')
    extract_subtitle(folder03,output_folder03, output_extension= ".srt")

def test_extract_sub_1_video():
    folder02 = Path(r"C:\Users\Heng2020\OneDrive\Python NLP\InputData\The Matrix Resurrections 2021.mkv")
    output_folder02 = Path(r'C:\Users\Heng2020\OneDrive\Python NLP\OutputData\SubtitleMatrix')
    extract_sub_1_video(folder02,output_folder02)




# @unittest.skipIf(should_skip, "Skipping: Test_extract_audio2s")
class Test_extract_audio2(unittest.TestCase):
    folder01 = Path(r"H:\D_Video\The Ark Season 01 Portuguese")
    French_bigbang = Path(r"E:\Videos\The Big Bang Theory\The Big Bang Theory French Season 06")
    output_folder = Path(r"E:\Videos\The Big Bang Theory\The Big Bang Theory French Season 06\Audio")
    # 4.82 min for 10 audios

    def test_basic01(self):
        extract_audio2(self.French_bigbang,self.output_folder,n_limit=10)
    
    def test_2_output_extension(self):
        pass
    pass

@unittest.skipIf(should_skip, "Skipping: Test_extract_1_subtitles")
class Test_extract_1_subtitle(unittest.TestCase):

    folder01 = Path(r"H:\D_Video\The Ark Season 01 Portuguese")
    video_name01 = "The Ark S01E02 PT.mkv"
    video_path01 = folder01 / video_name01
    output_folder01 = Path(r'H:\D_Video\The Ark Season 01 Portuguese\Subtitles')
    output_name = 'The Ark S01E02 PT'

    def test_basic01(self):
        actual = extract_sub_1_video(self.video_path01,self.output_folder01,self.output_name,output_extension='ass')
        expect = None
        self.assertEqual(expect, None)
    def test_basic02_no_outputname_and_ext(self):
        actual = extract_sub_1_video(self.video_path01,self.output_folder01)
        expect = None
        self.assertEqual(expect, None)
    
    def test_basic03_no_outputName(self):
        actual = extract_sub_1_video(self.video_path01,self.output_folder01,output_extension='.srt')
        expect = None
        self.assertEqual(expect, None)


        



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



if __name__ == '__main__':
    test_extract_subtitle()
    # test_extract_sub_1_video()
    # unittest.main()