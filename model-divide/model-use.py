from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import string
import re
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pandas as pd
import json
from keras.models import load_model
import numpy as np

# 创建类别索引到状态名称的映射
index_to_label = {0: 'no problem', 1: 'may problem', 2: 'anxiety', 3: 'stress', 4: 'ptsd'}

# 创建词干提取对象
stemmer = PorterStemmer()

# 定义清理文本的函数
def clean_text(text):
    # 替换错误的编码字符
    text = text.replace('鈥檚', "'s").replace('鈥檝e', "'ve").replace('鈥檛', "n't")
    # 使用正则表达式去除其他可能的错误字符
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    # 去除多余的空格
    text = ' '.join(text.split())
    return text

# 定义预处理函数
def preprocess_text(text):
    # 清理文本
    text = clean_text(text)
    # 去除数字
    text = re.sub(r'\d+', '', text)
    # 去除非英文字符
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    # 转换为小写
    text = text.lower()
    # 去除标点符号
    text = text.translate(str.maketrans('', '', string.punctuation))
    # 分词
    words = word_tokenize(text)
    # 词干提取并去除停用词
    words = [stemmer.stem(word) for word in words if word not in ENGLISH_STOP_WORDS]
    # 去除多余的空格
    words = [word for word in words if word.strip()]
    return ' '.join(words)

# 读取训练数据集
df_train = pd.read_csv('dreaddit-train.csv')

# 填充缺失值
df_train['text'] = df_train['text'].fillna('')

# 对文本数据进行预处理
df_train['text'] = df_train['text'].apply(preprocess_text)

# 将'Assistance'，'Relationships'，'Homeless'，'Almosthomeless'这几个标签都替换为'noproblem'
df_train['subreddit'] = df_train['subreddit'].replace(['assistance', 'relationships', 'homeless', 'almosthomeless'], 'no problem')
# df_train['subreddit'] = df_train['subreddit'].replace(['anxiety', 'stress', 'ptsd'], 'have problem')
df_train['subreddit'] = df_train['subreddit'].replace(['domesticviolence', 'survivorsofabuse'], 'may problem')

#df_train = df_train[df_train['subreddit'] != 'food_pantry']

# 删除'may problem'类的所有样本
df_train = df_train[df_train['subreddit'] != 'may problem']

# 使用Tokenizer将训练数据集的文本数据转换为特征向量
max_fatures = 3000
tokenizer = Tokenizer(num_words=max_fatures, split=' ')
tokenizer.fit_on_texts(df_train['text'].values)
X_train = tokenizer.texts_to_sequences(df_train['text'].values)
X_train = pad_sequences(X_train)

# 获取词汇表
word_index = tokenizer.word_index

# 将词汇表保存为json文件
with open('my_model4_word_index.json', 'w') as f:
    json.dump(word_index, f)

# # 加载模型
# model = load_model('my_model5.h5')
#
# # 加载词汇表
# with open('my_model5_word_index.json', 'r') as f:
#     word_index = json.load(f)
#
# # 创建一个新的Tokenizer
# tokenizer = Tokenizer()
# tokenizer.word_index = word_index
#
# # 使用Tokenizer将新的文本数据转换为序列
# new_text = "some new text data"
# new_text = preprocess_text(new_text)  # 使用与训练时相同的预处理步骤
# sequences = tokenizer.texts_to_sequences([new_text])
# sequences = pad_sequences(sequences, maxlen=114)  # 使用与训练时相同的序列长度
#
# # 使用模型进行预测
# predictions = model.predict(sequences)
#
# # 获取预测类别
# predicted_classes = np.argmax(predictions, axis=1)
#
# # 使用index_to_label字典将类别索引转换为状态名称
# predicted_labels = [index_to_label[i] for i in predicted_classes]
#
# # 打印预测的状态名称
# print(predicted_labels)