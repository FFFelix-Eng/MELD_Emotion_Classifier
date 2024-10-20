import os
import csv
from moviepy.editor import VideoFileClip


def extract_audio_from_videos(directory):
    error_files = []  # 用于记录处理失败的文件

    # 遍历目录中的所有文件
    for filename in os.listdir(directory):
        if filename.endswith(".mp4"):  # 只处理 mp4 文件
            video_path = os.path.join(directory, filename)

            # 打印正在处理的文件名
            print(f"Processing {filename}...")

            try:
                # 加载视频文件
                video_clip = VideoFileClip(video_path)

                # 提取音频并保存
                audio_path = os.path.join(directory, filename.replace(".mp4", ".wav"))
                video_clip.audio.write_audiofile(audio_path)

                # 关闭视频文件
                video_clip.close()

                print(f"Saved audio as {audio_path}")

            except Exception as e:
                print(f"Failed to process {filename}: {e}")
                error_files.append(filename)  # 将报错的文件记录下来

    # 将报错的文件保存到 CSV 文件中
    if error_files:
        csv_file_path = os.path.join(directory, 'error_files.csv')
        with open(csv_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Filename', 'Error'])
            for error_file in error_files:
                writer.writerow([error_file, "Processing failed"])

        print(f"Error files saved to {csv_file_path}")
    else:
        print("All files processed successfully!")


# 指定你要处理的目录路径
directory_path = "../../MELD_Raw/dev/dev_splits"  # 替换为你的视频所在目录

# 调用函数提取音频
extract_audio_from_videos(directory_path)

