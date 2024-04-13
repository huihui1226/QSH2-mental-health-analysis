from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
import string
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.tokenize import word_tokenize
import joblib
from nltk.stem import PorterStemmer
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer

# 创建词干提取对象
stemmer = PorterStemmer()

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
df_train['subreddit'] = df_train['subreddit'].replace(['anxiety', 'stress', 'ptsd'], 'have problem')
df_train['subreddit'] = df_train['subreddit'].replace(['domesticviolence', 'survivorsofabuse'], 'may problem')

df_train = df_train[df_train['subreddit'] != 'food_pantry']
#df_train = df_train[df_train['subreddit'] != 'may problem']

# 使用TfidfVectorizer将训练数据集的文本数据转换为特征向量
vectorizer = TfidfVectorizer(max_features=3000)
X_train = vectorizer.fit_transform(df_train['text'].values).toarray()

# 将目标变量转换为分类编码
y_train = pd.get_dummies(df_train['subreddit']).values

# 划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

# 创建决策树模型
model = DecisionTreeClassifier()

# 定义要搜索的参数网格
param_grid = {
    'max_depth': [220, 230, 240, 250],
    'min_samples_split': [1,2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}

# 创建GridSearchCV对象
grid_search = GridSearchCV(model, param_grid, cv=5, verbose=1)

# 训练模型
grid_search.fit(X_train, y_train)

# 打印最佳参数
print('Best parameters: ', grid_search.best_params_)

# 使用最佳参数创建新的模型
best_model = grid_search.best_estimator_

# 预测验证集结果
y_pred = best_model.predict(X_val)

# 计算准确率
accuracy = accuracy_score(y_val, y_pred)
print('Validation Accuracy: ', accuracy)

# 读取测试数据集
df_test = pd.read_csv('dreaddit-test.csv')

# 填充缺失值
df_test['text'] = df_test['text'].fillna('')

# 将测试数据集中的'Assistance'，'Relationships'，'Homeless'，'Almosthomeless'这几个标签都替换为'noproblem'
df_test['subreddit'] = df_test['subreddit'].replace(['assistance', 'relationships', 'homeless', 'almosthomeless'], 'no problem')
df_test['subreddit'] = df_test['subreddit'].replace(['anxiety', 'stress', 'ptsd'], 'have problem')
df_test['subreddit'] = df_test['subreddit'].replace(['domesticviolence', 'survivorsofabuse'], 'may problem')

df_test = df_test[df_test['subreddit'] != 'food_pantry']
#df_test = df_test[df_test['subreddit'] != 'may problem']

# 对文本数据进行预处理
df_test['text'] = df_test['text'].apply(preprocess_text)

# 使用之前训练好的Vectorizer将测试数据集的文本数据转换为特征向量
X_test = vectorizer.transform(df_test['text'].values).toarray()

# 使用之前训练好的模型进行预测
predictions_test = best_model.predict(X_test)

# 如果测试数据集中包含真实的标签，可以计算准确率
accuracy_test = accuracy_score(pd.get_dummies(df_test['subreddit']).values.argmax(axis=1), predictions_test.argmax(axis=1))
print(f'测试数据集的模型准确率: {accuracy_test:.3f}')

# 如果准确率大于60%，则保存模型
if accuracy_test > 0.60:
    joblib.dump(best_model, f'DecisionTree-model-{accuracy_test:.3f}.pkl')