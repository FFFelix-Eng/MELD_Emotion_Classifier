import os
import torchaudio
import torch
from scipy import signal


# 合并双声道为单声道
def merge_stereo_to_mono(waveform):
    # 如果是双声道，取两个通道的平均值合并为单声道
    if waveform.size(0) == 2:  # 检查是否为双声道
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    return waveform


# 插值函数，将音频数据缩放到目标长度
def resample_to_fixed_length(waveform, target_length):
    # 将 PyTorch tensor 转换为 numpy 数组
    waveform_np = waveform.squeeze().numpy()  # 变成一维数组
    # 使用 SciPy 的信号重采样函数进行插值
    resampled_waveform = signal.resample(waveform_np, target_length)
    # 将结果转换回 PyTorch tensor
    return torch.tensor(resampled_waveform).unsqueeze(0)  # 加上通道维度


# 处理和保存音频
def process_and_save_audio(file_path, output_folder):
    # 加载音频文件
    waveform, sample_rate = torchaudio.load(file_path)

    # 重采样到16kHz
    if sample_rate != 16000:
        resample_transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resample_transform(waveform)

    # 合并声道为单声道
    waveform = merge_stereo_to_mono(waveform)

    # 目标采样点数（3秒 * 16000采样率 = 48000个采样点）
    target_num_samples = 16000 * 3

    # 插值到目标长度
    waveform = resample_to_fixed_length(waveform, target_num_samples)

    # 保存处理后的音频文件
    output_file_path = os.path.join(output_folder, os.path.basename(file_path))
    torchaudio.save(output_file_path, waveform, 16000)


# 处理音频文件夹
def process_audio_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历文件夹中的所有WAV文件
    for audio_file in os.listdir(input_folder):
        if audio_file.endswith('.wav'):
            file_path = os.path.join(input_folder, audio_file)
            process_and_save_audio(file_path, output_folder)
            print(f"处理并保存文件: {audio_file}")


# 使用示例
input_folder = f'../MELD_Raw/dev/dev_splits/aud_raw/'
output_folder = f'../MELD_Raw/dev/dev_splits/aud_resampled/'
process_audio_folder(input_folder, output_folder)
