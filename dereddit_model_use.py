import string
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from joblib import load
from sklearn.feature_extraction.text import TfidfVectorizer

# 创建词形还原对象
lemmatizer = WordNetLemmatizer()

# 加载模型和词汇表
model = load('dereddit_model_update.joblib')
vectorizer = load('dereddit_vocabulary_update.joblib')

# 定义预处理函数
def preprocess_text(text):
    # 转换为小写
    text = text.lower()
    # 去除标点符号
    text = text.translate(str.maketrans('', '', string.punctuation))
    # 分词
    words = word_tokenize(text)
    # 词形还原并去除停用词
    words = [lemmatizer.lemmatize(word) for word in words if word not in ENGLISH_STOP_WORDS]
    return ' '.join(words)

# 从控制台获取输入句子
sentence = input("请输入你的句子：")

# 预处理输入句子
sentence = preprocess_text(sentence)

# 将输入句子转换为特征向量
sentence_vector = vectorizer.transform([sentence])

# 使用模型进行预测
prediction = model.predict(sentence_vector)

# 输出预测结果
print(f'预测的mental health状态: {prediction[0]}')

