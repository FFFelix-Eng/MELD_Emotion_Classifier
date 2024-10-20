import os
import torchaudio


# 获取音频文件的时长
def get_audio_duration(file):
    waveform, sample_rate = torchaudio.load(file)
    duration = waveform.size(1) / sample_rate  # 时长 = 采样点数 / 采样率
    return duration


# 统计文件夹中所有音频文件的时长
def calculate_durations(audio_folder):
    durations = []

    for audio_file in os.listdir(audio_folder):
        file_path = os.path.join(audio_folder, audio_file)

        # 检查文件是否为音频文件
        if os.path.isfile(file_path) and file_path.endswith('.wav'):
            try:
                duration = get_audio_duration(file_path)
                durations.append(duration)
            except Exception as e:
                print(f"无法加载文件 {file_path} : {e}")

    if len(durations) > 0:
        avg_duration = sum(durations) / len(durations)
        print(f"平均时长: {avg_duration:.2f} 秒")
        print(f"最大时长: {max(durations):.2f} 秒")
        print(f"最小时长: {min(durations):.2f} 秒")
    else:
        print("未找到任何音频文件")


# 文件夹路径
audio_folder = '../MELD_Raw/train/train_splits/aud_raw/'

# 调用函数统计时长
calculate_durations(audio_folder)
