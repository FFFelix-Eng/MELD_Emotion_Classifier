import os.path
import torchaudio

# 加载音频文件
path = f'../MELD_Raw/train/train_splits/aud_raw/'
audio_file = "dia0_utt0.wav"
file = os.path.join(path, audio_file)

print(os.path.exists(file))

# 使用 torchaudio 加载音频
waveform, sample_rate = torchaudio.load(file)

print(f"采样率: {sample_rate} Hz")
print(f"采样点数: {waveform.size(1)}")  # 打印采样点的数量
print(f"通道数: {waveform.size(0)}")
print(f"数据类型:{waveform.dtype}")