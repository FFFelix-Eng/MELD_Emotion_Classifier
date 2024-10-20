import pandas as pd

# 读取CSV文件
df = pd.read_csv('../../data/text2/dev.csv')

# 删除符合条件的行：dialogue_ID = 110 且 utter_id 为 7、8、9 的行
df = df[~((df['Dialogue_ID'] == 110) & (df['Utterance_ID'].isin([7, 8, 9])))]

# 保存修改后的CSV文件
df.to_csv('../../data/text2/file_modified.csv', index=False)

print("删除成功并保存到 file_modified.csv")
