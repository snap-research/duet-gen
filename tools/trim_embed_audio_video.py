from moviepy.editor import VideoFileClip, AudioFileClip
from pydub import AudioSegment
from io import BytesIO
import os
import pickle
import glob

def trim_audio_in_memory(input_audio_path, start_time, end_time, output_audio_path = 'trimm.mp3'):
    # Load the audio file
    audio = AudioSegment.from_mp3(input_audio_path)    
    # Convert start and end times from seconds to milliseconds
    start_time_ms = start_time * 1000
    end_time_ms = end_time * 1000
    # Trim the audio file
    trimmed_audio = audio[start_time_ms:end_time_ms]
    trimmed_audio.export(output_audio_path, format="mp3")

    
def embed_audio_on_video(video_path, audio_path, output_path):
    # Load the video file
    video = VideoFileClip(video_path)
    # Load the trimmed audio file
    if audio_path.endswith('.mp3'):
        audio = AudioFileClip(audio_path)
    elif audio_path.endswith('.mp4'):
        audiovideofile = VideoFileClip(audio_path)
        audio = audiovideofile.audio
    
    # Set the trimmed audio as the video’s audio
    video_with_audio = video.set_audio(audio)
    # Export the final video with the new audio
    video_with_audio.write_videofile(output_path, codec="libx264", audio_codec="aac")
    print(f"Video with new audio saved to: {output_path}")

if __name__ == '__main__':
    
    input_music_file = os.path.join('data', 'DD100', '400_100', 'test_music')
    input_video_filepath = '' # path to the generated mp4 files
    all_music_files = sorted(glob.glob(input_music_file + '/*.pkl'))
    all_video_files = glob.glob(input_video_filepath + '/*.mp4')
    for iter, video_path in enumerate(all_video_files):
        video_iter = int(video_path.split('\\')[-1].split('.')[0].split('_')[0])
        music_file_path = all_music_files[video_iter]
        with open(music_file_path, 'rb') as f:
            music_data_dict = pickle.load(f)
        input_audio_path = music_data_dict['music_sequence_name']
        start_time = music_data_dict['start_time']
        end_time = music_data_dict['stop_time']
        output_video_path = video_path[:-4] + '_music.mp4'
        if video_path.endswith('music.mp4') or os.path.exists(output_video_path):
            print(output_video_path +" exists")
        else:
            # Step 1: Trim the audio in memory
            trimmed_audio_path = 'trim.mp3'
            trim_audio_in_memory(input_audio_path, start_time, end_time, trimmed_audio_path)
            # Step 2: Embed the trimmed audio onto the video
            embed_audio_on_video(video_path, trimmed_audio_path, output_video_path)