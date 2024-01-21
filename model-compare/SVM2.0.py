import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from joblib import dump
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import string
import re
from sklearn import svm

# # 创建词形还原对象
# lemmatizer = WordNetLemmatizer()

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
    # # 词形还原并去除停用词
    # words = [lemmatizer.lemmatize(word) for word in words if word not in ENGLISH_STOP_WORDS]
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

# 使用TfidfVectorizer将训练数据集的文本数据转换为特征向量
# 使用n-grams
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
X_train_vectors = vectorizer.fit_transform(df_train['text'])

# 创建SVM模型
model = svm.SVC()

# 定义参数网格
param_grid = {
    'C': [1, 10, 15, 20, 25, 30],
    'gamma': [0.01, 0.1, 1, 10],
    'kernel': ['linear', 'rbf', ] #'poly', 'sigmoid'
}

# 创建网格搜索对象
grid_search = GridSearchCV(model, param_grid, cv=5)

# 训练模型并搜索最优参数
grid_search.fit(X_train_vectors, df_train['subreddit'])

# 输出最优参数
print(f'最优参数: {grid_search.best_params_}')

# 保存模型
dump(grid_search.best_estimator_, 'dereddit_model_SVM.joblib')

# 保存词汇表
dump(vectorizer, 'dereddit_vocabulary_SVM.joblib')

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

# 使用之前训练好的TfidfVectorizer将测试数据集的文本数据转换为特征向量
X_test_vectors = vectorizer.transform(df_test['text'])

# 使用之前训练好的模型进行预测
predictions_test = grid_search.best_estimator_.predict(X_test_vectors)

# 如果测试数据集中包含真实的标签，可以计算准确率
# 假设测试数据集中包含名为'subreddit'的列，其中包含真实的标签
accuracy_test = accuracy_score(df_test['subreddit'], predictions_test)
print(f'测试数据集的模型准确率: {accuracy_test:.2f}')
