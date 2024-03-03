
import pysrt
import pandas as pd
import os
from pydub import AudioSegment
from time import time
from playsound import playsound
from packaging import version

# Parse the SRT subtitle file

# srt_path = r"H:\D_Video\Westworld Portugues 04\Eng Sub\Westworld.S04E01 EngSub 02.srt"
# sub_output_name = 'Westworld_S04E01_EN02.xlsx'

srt_folder_path = r"H:\D_Video\The Ark Season 01 Portuguese\Subtitles\srt"
output_folder = r"H:\D_Video\The Ark Season 01 Portuguese\Subtitles\Excel generated"

alarm_path = r"H:\D_Music\Sound Effect positive-logo-opener.wav"

# sub_output = os.path.join(srt_folder_path,sub_output_name)
# what if I extract str directly from video .mvk?
def srt_to_df(srt_path):
    if ".srt" in srt_path:
        # 1 file case
        subs = pysrt.open(srt_path)
        # Initialize empty lists for storing data
        sentences = []
        start_times = []
        end_times = []
    
        # Extract data from each subtitle sentence
        for sub in subs:
            sentences.append(sub.text)
            start_times.append(sub.start.to_time())
            end_times.append(sub.end.to_time())
        
        clean_sentences = list(map(clean_subtitle,sentences))
    
        # Create a DataFrame
        df = pd.DataFrame({
            'sentence': clean_sentences,
            'start': start_times,
            'end': end_times
        })
        return df
    else:
        # many srt's file using folder
        str_file_names = get_full_filename(srt_path,".srt")
        df_list = []
        for str_file_name in str_file_names:
            each_df = srt_to_df(str_file_name)
            df_list.append(each_df)
        return df_list


def srt_to_csv(srt_path,output_path,encoding='utf-8-sig',index=False):
    # output should be total_path
    df_sub = srt_to_df(srt_path)
    # encoding='utf-8-sig' for Portuguese
    df_sub.to_csv(output_path, encoding=encoding,index=index)

def srt_to_Excel(srt_path,output_path,encoding='utf-8-sig',index=True,alarm_path = r"H:\D_Music\Sound Effect positive-logo-opener.wav"):
    
    # when .ass is converted to .srt, when I checked using notepad ++
    # it also has weird tag at the front
    # but seems like it's normal when export to Excel or even play using VLC Player 

    #TOFIX the Excel files created still give me an error:
        #This file can't be previewed because of an error in the Microsoft Excel previewer
    #But files are created just fine.
    
    # output should be total_path
    # srt_path could be folder
    
    # just upgrade it to support multiple files
    # it took about 1 hr on Sep 16, 2023
    from playsound import playsound
    
    if ".srt" in srt_path:
        # 1 srt file
        df_sub = srt_to_df(srt_path)
        # encoding='utf-8-sig' for Portuguese
        df_sub.to_excel(output_path, encoding=encoding,index=index)
    else:
        str_full_names = get_full_filename(srt_path,".srt")
        str_file_names = get_filename(srt_path,".srt")
        
        t_df = []
        t_write_excel = []
        
        for i, str_full_name in enumerate(str_full_names):
            
            ts_df_start = time()
            ################### main tested function
            each_df = srt_to_df(str_full_name)
            # shift the index of each_df to start at 1
            each_df.index += 1
            #---------------------------
            ts_df_end = time()
            t_df_duration = ts_df_end - ts_df_start
            t_df.append(t_df_duration)
            
            
            xlsx_name = str_file_names[i].replace(".srt",".xlsx")
            
            output_full = os.path.join(output_path,xlsx_name)
            
            
            ts_write_start = time()
            ################### main tested function
            if version.parse(pd.__version__) < version.parse("2.0.0"):
                each_df.to_excel(output_full, encoding=encoding,index=index)
            else:
                # version 2 of pandas onwards doesn't have encoding
                each_df.to_excel(output_full,index=index)
            #------------------------------
            ts_write_end = time()
            t_write_duration = ts_write_end - ts_write_start
            t_write_excel.append(t_write_duration)
            
            del each_df
        
        total_df = sum(t_df)
        total_write = sum(t_write_excel)
        
        print("*"*30)
        print("All srt files are converted to Excel sucessfully!!!")
        print(f"Time converting to df: {total_df: .2f} sec")
        print(f"Time writing to Excel: {total_write: .2f} sec")
        # print(t_write_excel)
        print("*"*30)
        playsound(alarm_path)

def to_ms(time_obj):
    time_obj_ms = (time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second) * 1000 + time_obj.microsecond // 1000
    return time_obj_ms

def get_filename(folder_path,extension = "all"):
    # also include "folder"  case
# tested small
    if extension == "all":
        out_list = [ file for file in os.listdir(folder_path) ]

    elif isinstance(extension,str):
        extension_temp = [extension]

        out_list = []

        for file in os.listdir(folder_path):
            if "." in file:
                file_extension = file.split('.')[-1]
                for each_extention in extension_temp:
                    # support when it's ".csv" or only "csv"
                    if file_extension in each_extention:
                        out_list.append(file)
            elif extension == "folder":
                out_list.append(file)


    elif isinstance(extension,list):
        out_list = []
        for file in os.listdir(folder_path):

            if "." in file:
                file_extension = file.split('.')[-1]
                for each_extention in extension:
                    # support when it's ".csv" or only "csv"
                    if file_extension in each_extention:
                        out_list.append(file)

            elif "folder" in extension:
                out_list.append(file)

        return out_list

    else:
        print("Don't support this dataype for extension: please input only string or list")
        return False

    return out_list

def get_full_filename(folder_path,extension = "all"):
    # tested small
    short_names = get_filename(folder_path,extension)
    out_list = []
    for short_name in short_names:
        full_name = os.path.join(folder_path,short_name)
        out_list.append(full_name)
    return out_list

def clean_subtitle(string):
    import re
    pattern1 = "<.*?>"
    
    pattern2 = "<\/[a-zA-Z]>"
    
    string1 = re.sub(pattern1, "", string)
    string2 = string1.replace("\n"," ")
    new_string = re.sub(pattern2,"",string2)
    return new_string



# srt_to_Excel(srt_path,sub_output)

srt_to_Excel(srt_folder_path,output_folder)
# df = srt_to_df(srt_path)
# df_list = srt_to_df(srt_folder_path)
# print(df_list[3].head())
# string_list = df['sentence'].tolist()
# test_str3 = string_list[24]

# test_str = string_list[240]
# test_str2 = "corrected byFOLLOW US: <font color=""#ff0000"">@LIO_OFFICIAL</font>"
# clean_str = clean_subtitle(test_str3)

# clean_list = list(map(clean_subtitle,string_list))


