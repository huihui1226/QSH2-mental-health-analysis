from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import string
import re
from keras.callbacks import EarlyStopping
from scikeras.wrappers import KerasClassifier
from keras.callbacks import ModelCheckpoint
import os
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Embedding, SpatialDropout1D, LSTM, Dropout, Dense, Bidirectional, GRU
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pandas as pd

from keras.layers import Conv1D, MaxPooling1D, BatchNormalization

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

# # 删除'may problem'类的所有样本
# df_train = df_train[df_train['subreddit'] != 'may problem']

# 使用Tokenizer将训练数据集的文本数据转换为特征向量
max_fatures = 3000
tokenizer = Tokenizer(num_words=max_fatures, split=' ')
tokenizer.fit_on_texts(df_train['text'].values)
X_train = tokenizer.texts_to_sequences(df_train['text'].values)
X_train = pad_sequences(X_train)

# 创建LSTM模型
embed_dim = 128
lstm_out = 128

batch_size = 32 # 定义batch_size

model = Sequential()
model.add(Embedding(max_fatures, embed_dim,input_length = X_train.shape[1])) #use word2Vec
model.add(Conv1D(128, 4, activation='relu')) # 添加CNN层
model.add(MaxPooling1D(pool_size=4)) # 添加池化层
model.add(BatchNormalization()) # 添加BatchNormalization层
#model.add(Dropout(0.2)) # 添加Dropout层
model.add(Bidirectional(LSTM(lstm_out,return_sequences=True))) # 添加return_sequences=True以便在下一层使用序列输出
#model.add(Dropout(0.2)) # 添加Dropout层
#model.add(BatchNormalization()) # 添加BatchNormalization层
model.add(Bidirectional(LSTM(lstm_out,return_sequences=True))) # 添加return_sequences=True以便在下一层使用序列输出
#model.add(Dropout(0.3)) # 添加Dropout层
model.add(LSTM(lstm_out)) # 添加第二个LSTM层
#model.add(GRU(lstm_out)) # 使用GRU替换LSTM
model.add(Dense(50, activation='relu')) # 添加Dense层
model.add(BatchNormalization()) # 添加BatchNormalization层
model.add(Dense(6,activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics = ['accuracy']) #    RMSprop(learning_rate=0.0005)

# 创建一个只保存最好模型的检查点
checkpoint = ModelCheckpoint('best3_model.h5', monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

# 添加早停
early_stop = EarlyStopping(monitor='val_loss', patience=3)
model.fit(X_train, pd.get_dummies(df_train['subreddit']).values, epochs = 10, batch_size=batch_size, verbose = 2, callbacks=[checkpoint, early_stop], validation_split=0.1) # 添加validation_split参数 early_stop,


# 读取测试数据集
df_test = pd.read_csv('dreaddit-test.csv')

# 填充缺失值
df_test['text'] = df_test['text'].fillna('')

# 将测试数据集中的'Assistance'，'Relationships'，'Homeless'，'Almosthomeless'这几个标签都替换为'noproblem'
df_test['subreddit'] = df_test['subreddit'].replace(['assistance', 'relationships', 'homeless', 'almosthomeless'], 'no problem')
# df_test['subreddit'] = df_test['subreddit'].replace(['anxiety', 'stress', 'ptsd'], 'have problem')
df_test['subreddit'] = df_test['subreddit'].replace(['domesticviolence', 'survivorsofabuse'], 'may problem')

# # 删除'may problem'类的所有样本
# df_test = df_test[df_test['subreddit'] != 'may problem']

# 对文本数据进行预处理
df_test['text'] = df_test['text'].apply(preprocess_text)

# 使用之前训练好的Tokenizer将测试数据集的文本数据转换为特征向量
X_test = tokenizer.texts_to_sequences(df_test['text'].values)
X_test = pad_sequences(X_test, maxlen=X_train.shape[1])

# 使用之前训练好的模型进行预测
predictions_test = model.predict(X_test)

# 如果测试数据集中包含真实的标签，可以计算准确率
accuracy_test = accuracy_score(pd.get_dummies(df_test['subreddit']).values.argmax(axis=1), predictions_test.argmax(axis=1))
print(f'测试数据集的模型准确率: {accuracy_test:.3f}')

# 如果准确率大于60%，则保存模型
if accuracy_test > 0.60:
    os.rename('best3_model.h5', f'LSTM-model-3Layer-{accuracy_test:.3f}.h5')
    # model.save(f'LSTM-model-3Layer-{accuracy_test:.3f}.h5')
