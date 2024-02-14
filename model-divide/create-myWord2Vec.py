from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import string
import re
from gensim.models import Word2Vec
import pandas as pd

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

# 将预处理后的文本转换为句子列表
sentences = df_train['text'].apply(lambda x: x.split())

# 训练Word2Vec模型
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

# vector_size：这是词向量的维度，即每个词将被表示为一个多少维的向量。一般来说，这个值可以设置在100-300之间，具体取决于你的数据集的大小。
#
# window：这是Word2Vec模型中的窗口大小。它表示当前词与预测词在一个句子中的最大距离。例如，如果window=5，那么模型将考虑当前词前后各5个词内的词作为上下文。这个值通常设置在5-10之间。
#
# min_count：这是模型中考虑的词的最小频率。如果一个词的频率低于min_count，那么这个词将被忽略。这可以帮助模型去除一些罕见的词。这个值通常设置在1-5之间。
#
# workers：这是训练模型的线程数。如果你的机器有多个核，那么增加这个值可以加速模型的训练。这个值通常设置为你的机器的核数

# 保存模型
model.save("word2vec.model")