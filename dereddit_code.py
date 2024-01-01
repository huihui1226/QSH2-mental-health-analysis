import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from joblib import dump
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import string

# 创建词形还原对象
lemmatizer = WordNetLemmatizer()

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

# 读取数据集
df = pd.read_csv('dreaddit-train.csv')

# 填充缺失值
df['text'] = df['text'].fillna('')

# 对文本数据进行预处理
df['text'] = df['text'].apply(preprocess_text)

# 分割数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['subreddit'], test_size=0.2, random_state=42)

# 使用TfidfVectorizer将文本数据转换为特征向量
# 使用n-grams
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)

# 创建逻辑回归模型
model = LogisticRegression(multi_class='multinomial', solver="newton-cg") #solver='saga', max_iter=5000

# 定义参数网格
param_grid = {
    'C': [0.1, 1, 10, 100],
    'penalty': ['l2']
}

# 创建网格搜索对象
grid_search = GridSearchCV(model, param_grid, cv=5)

# 训练模型并搜索最优参数
grid_search.fit(X_train_vectors, y_train)

# 输出最优参数
print(f'最优参数: {grid_search.best_params_}')

# 使用最优参数的模型进行预测
predictions = grid_search.predict(X_test_vectors)

# 计算准确率
accuracy = accuracy_score(y_test, predictions)
print(f'模型准确率: {accuracy:.2f}')

# 保存模型
dump(grid_search.best_estimator_, 'dereddit_model.joblib')

# 保存词汇表
dump(vectorizer, 'dereddit_vocabulary.joblib')
