import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

# 首先，确保您已经下载了nltk的停用词集
nltk.download('stopwords')
nltk.download('punkt')

# 读取数据集
df = pd.read_csv('dreaddit-train.csv')

# 获取英文停用词集
stop_words = set(stopwords.words('english'))

# 确保df.text中的所有项都是字符串类型，并处理NaN值
df['subreddit'] = df['subreddit'].fillna('').apply(str)

# 提取'text'列的文本，并进行分词
words = word_tokenize(" ".join(review for review in df.subreddit))

# 移除停用词和非字母字符
filtered_words = [word for word in words if word.isalpha() and word not in stop_words]

# 将处理后的单词合并为一个长字符串
cleaned_text = " ".join(filtered_words)

# 创建词云对象
wordcloud = WordCloud(width = 800, height = 400, background_color ='white').generate(cleaned_text)

# 显示生成的词云
plt.figure(figsize = (20, 10), facecolor = None)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.tight_layout(pad = 0)


# 在屏幕上显示图像
plt.show()
