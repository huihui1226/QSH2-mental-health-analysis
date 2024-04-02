import tkinter as tk
from tkinter import filedialog, messagebox
import json
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import pandas as pd
import numpy as np
from transformers import TFBertForSequenceClassification
import joblib

# # 创建类别索引到状态名称的映射
# index_to_label = {0: 'no problem', 1: 'may problem', 2: 'anxiety', 3: 'stress', 4: 'ptsd'}

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
    words = [stemmer.stem(word) for word in words if word not in stopwords.words('english')]
    # # 去除不在词汇表中的词
    # words = [word for word in words if word in current_tokenizer.word_index]
    # 去除多余的空格
    words = [word for word in words if word.strip()]
    return ' '.join(words)


# 加载模型和词汇表的函数
def load_model_and_vocab(model_name, maxlen=None, index_to_label=None):
    if model_name == 'my_model4':
        model = joblib.load(model_name + '.joblib')
        tokenizer = joblib.load(model_name + '_vocabulary.joblib')
    else:
        model = load_model(model_name + '.h5')
        with open(model_name + '_word_index.json', 'r') as f:
            word_index = json.load(f)
        tokenizer = Tokenizer()
        tokenizer.word_index = word_index
    return model, tokenizer, maxlen, index_to_label

# 创建一个字典，键是模型的名称，值是对应的模型和词汇表
models = {
    'my_model5': load_model_and_vocab('my_model5', 114, {0: 'no problem', 1: 'may problem', 2: 'anxiety', 3: 'stress', 4: 'ptsd'}),
    'my_model2': load_model_and_vocab('my_model2', 111, {0: 'no problem', 1: 'have problem'}),
    'my_model4': load_model_and_vocab('my_model4') #, 111, {0: 'no problem', 1:  'anxiety', 2: 'stress', 3: 'ptsd', 4: 'food_pantry'}
}

# 创建一个变量来保存当前选择的模型和词汇表
current_model, current_tokenizer, current_maxlen, current_index_to_label = models['my_model4']

# 修改predict函数，使用current_model和current_tokenizer
def predict():
    # 获取文本框中的内容
    new_text = entry.get("1.0", tk.END)  # 从Text控件获取文本
    # 检查文本框是否为空
    if not new_text.strip():
        messagebox.showerror("错误", "文本框不能为空")
        return
    # 对新的文本数据进行预处理
    new_text = preprocess_text(new_text)
    # 使用current_tokenizer将新的文本数据转换为特征向量
    if model_var.get() == 'my_model4':
        features = current_tokenizer.transform([new_text])
        # 使用current_model进行预测
        predictions = current_model.predict(features)
        result_label.config(text="预测结果: " + str(predictions[0]))
    else:
        sequences = current_tokenizer.texts_to_sequences([new_text])
        sequences = pad_sequences(sequences, maxlen=current_maxlen)
        # 使用current_model进行预测
        predictions = current_model.predict(sequences)
        # 获取预测类别
        predicted_classes = np.argmax(predictions, axis=1)
        # 使用current_index_to_label字典将类别索引转换为状态名称
        predicted_labels = [current_index_to_label[i] for i in predicted_classes]
        # 在标签上显示预测结果
        result_label.config(text="预测结果: " + predicted_labels[0])

# 创建一个函数来读取文件并进行预测
def predict_file():
    # 打开文件选择对话框
    file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
    if not file_path:
        return
    # 读取文件内容
    with open(file_path, 'r') as f:
        lines = f.read().split('\n')  # 使用回车键分割文本
    results = []
    for new_text in lines:
        # 对新的文本数据进行预处理
        new_text = preprocess_text(new_text)
        # 使用current_tokenizer将新的文本数据转换为特征向量
        if model_var.get() == 'my_model4':
            features = current_tokenizer.transform([new_text])
            # 使用current_model进行预测
            predictions = current_model.predict(features)
            result = str(predictions[0])
        else:
            sequences = current_tokenizer.texts_to_sequences([new_text])
            sequences = pad_sequences(sequences, maxlen=current_maxlen)
            # 使用current_model进行预测
            predictions = current_model.predict(sequences)
            # 获取预测类别
            predicted_classes = np.argmax(predictions, axis=1)
            # 使用current_index_to_label字典将类别索引转换为状态名称
            predicted_labels = [current_index_to_label[i] for i in predicted_classes]
            # 在标签上显示预测结果
            result = predicted_labels[0]
        results.append(result)
    # 将结果保存到result.txt中
    with open('result.txt', 'w') as f:
        for result in results:
            f.write(result + '\n')


# 创建主窗口
root = tk.Tk()
root.title("文本情感预测")

# 在GUI中添加一个下拉菜单来选择模型
model_var = tk.StringVar(root)
model_var.set('my_model4')  # 设置初始值
model_option = tk.OptionMenu(root, model_var, *models.keys())
model_option.pack()


# 当选择的模型改变时，更新current_model和current_tokenizer
def on_model_change(*args):
    global current_model, current_tokenizer, current_maxlen, current_index_to_label
    current_model, current_tokenizer, current_maxlen, current_index_to_label = models[model_var.get()]
model_var.trace('w', on_model_change)


# 创建一个文本输入框
entry = tk.Text(root, width=50, height=10)  # 设置高度为10行
entry.pack()
# # 创建一个文本输入框
# entry = tk.Entry(root, width=50)
# entry.pack()

# 创建一个按钮，点击时调用predict函数
button = tk.Button(root, text="forecast", command=predict)
button.pack()

# 创建一个新的按钮，点击时调用predict_file函数
button_file = tk.Button(root, text="Prediction from file", command=predict_file)
button_file.pack()

# 创建一个标签来显示预测结果
result_label = tk.Label(root, text="Predicted result: ")
result_label.pack()

# 运行主循环
root.mainloop()