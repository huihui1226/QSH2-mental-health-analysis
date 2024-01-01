import pandas as pd
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, Embedding, Flatten
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from joblib import dump
from nltk.tokenize import word_tokenize
import re
import string
from joblib import load
from keras.models import load_model
from sklearn.metrics import accuracy_score
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.layers import Dropout

# 创建词干提取对象
stemmer = PorterStemmer()

# 定义预处理函数
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

# 将标签编码为整数
label_encoder = LabelEncoder()
df_train['subreddit'] = label_encoder.fit_transform(df_train['subreddit'].replace(['assistance', 'relationships', 'homeless', 'almosthomeless'], 'noproblem'))

# 使用Tokenizer将文本数据转换为序列
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df_train['text'])
X_train_sequences = tokenizer.texts_to_sequences(df_train['text'])

# 对序列进行填充以确保它们具有相同的长度
max_sequence_length = max([len(seq) for seq in X_train_sequences])
X_train_padded = pad_sequences(X_train_sequences, maxlen=max_sequence_length)

# 将标签转换为分类格式
y_train_categorical = to_categorical(df_train['subreddit'])

# 创建早停法回调函数
early_stopping = EarlyStopping(monitor='val_loss', patience=3)

# 创建神经网络模型
model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=100, input_length=max_sequence_length))
model.add(Flatten())
model.add(Dense(512, activation='relu'))  # 增加神经元数量
model.add(Dropout(0.5))  # 添加Dropout层
model.add(Dense(256, activation='relu'))  # 添加额外的Dense层
model.add(Dropout(0.5))  # 添加Dropout层
model.add(Dense(y_train_categorical.shape[1], activation='softmax'))
# 创建一个Adam优化器对象，并设置学习率
optimizer = Adam(learning_rate=0.001)

# 编译模型
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 在模型训练时添加回调函数
model.fit(X_train_padded, y_train_categorical, epochs=5, batch_size=32, validation_split=0.1, callbacks=[early_stopping])

# 保存模型和Tokenizer
model.save('dreaddit_model_nn.keras')
dump(tokenizer, 'dreaddit_tokenizer.joblib')
dump(label_encoder, 'dreaddit_label_encoder.joblib')

# 读取测试数据集
df_test = pd.read_csv('dreaddit-test.csv')

# 填充缺失值
df_test['text'] = df_test['text'].fillna('')

# 对文本数据进行预处理
df_test['text'] = df_test['text'].apply(preprocess_text)

# 加载之前保存的Tokenizer和LabelEncoder
tokenizer = load('dreaddit_tokenizer.joblib')
label_encoder = load('dreaddit_label_encoder.joblib')

# 将测试数据集的文本数据转换为序列
X_test_sequences = tokenizer.texts_to_sequences(df_test['text'])

# 对序列进行填充
X_test_padded = pad_sequences(X_test_sequences, maxlen=max_sequence_length)

# 加载模型
model = load_model('dreaddit_model_nn.keras')

# 使用模型进行预测
predictions_test = model.predict(X_test_padded)

# 将预测结果转换为标签
predicted_labels = label_encoder.inverse_transform(predictions_test.argmax(axis=1))

# 如果测试数据集中包含真实的标签，可以计算准确率
df_test['subreddit'] = label_encoder.transform(df_test['subreddit'].replace(['assistance', 'relationships', 'homeless', 'almosthomeless'], 'noproblem'))
accuracy_test = accuracy_score(df_test['subreddit'], predicted_labels)
print(f'测试数据集的模型准确率: {accuracy_test:.2f}')