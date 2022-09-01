# Python 3 code to split videos into 10sec videos
# Import necessary modules
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy.editor import *
import os
# Get the path for present directory
os.getcwd()

# Get the list of videos present in the specified directory
l= os.listdir('D:\\voilance detection project\\Trail codes\\cliping videos')
for vid in l:
# Consider only .mp4 files
  if(vid[-4:]=='.mp4'): 
    required_video_file = vid
    clip = VideoFileClip(required_video_file)
    duration = clip.duration
    duration = int(duration//10)
    for time in range(duration):
      starttime = int(time*10)
      endtime = int((time+1)*10)
      clips = clip.subclip(starttime,endtime)
      clips.write_videofile(str(time+1)+vid)
