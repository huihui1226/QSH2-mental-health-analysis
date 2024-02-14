import pandas as pd
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, TFBertForSequenceClassification
from transformers import InputExample, InputFeatures
import tensorflow as tf
from sklearn.metrics import accuracy_score

# 创建词干提取对象
stemmer = PorterStemmer()

# 定义清理文本的函数
def clean_text(text):
    text = text.replace('鈥檚', "'s").replace('鈥檝e', "'ve").replace('鈥檛', "n't")
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = ' '.join(text.split())
    return text

# 定义预处理函数
def preprocess_text(text):
    text = clean_text(text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = word_tokenize(text)
    words = [stemmer.stem(word) for word in words if word not in stopwords.words('english')]
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
df_train['subreddit'] = df_train['subreddit'].replace(['domesticviolence', 'survivorsofabuse'], 'may problem')

# # 删除'may problem'类的所有样本
# df_train = df_train[df_train['subreddit'] != 'may problem']

# 使用LabelEncoder将标签转换为数字
le = LabelEncoder()
df_train['subreddit'] = le.fit_transform(df_train['subreddit'])

# 创建tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# 创建模型
model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=6) # 根据你的类别数量修改num_labels

# 对文本数据进行预处理
def convert_data_to_examples(df, DATA_COLUMN, LABEL_COLUMN):
  InputExamples = df.apply(lambda x: InputExample(guid=None,
                                                  text_a = x[DATA_COLUMN],
                                                  text_b = None,
                                                  label = x[LABEL_COLUMN]), axis = 1)
  return InputExamples

def convert_examples_to_tf_dataset(examples, tokenizer, max_length=128):
    features = []

    for e in examples:
        input_dict = tokenizer.encode_plus(
            e.text_a,
            add_special_tokens=True,
            max_length=max_length,
            return_token_type_ids=True,
            return_attention_mask=True,
            pad_to_max_length=True,
            truncation=True
        )

        input_ids, token_type_ids, attention_mask = (input_dict["input_ids"],
            input_dict["token_type_ids"], input_dict['attention_mask'])

        features.append(
            InputFeatures(
                input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, label=e.label
            )
        )

    def gen():
        for f in features:
            yield (
                {
                    "input_ids": f.input_ids,
                    "attention_mask": f.attention_mask,
                    "token_type_ids": f.token_type_ids,
                },
                f.label,
            )

    return tf.data.Dataset.from_generator(
        gen,
        ({"input_ids": tf.int32, "attention_mask": tf.int32, "token_type_ids": tf.int32}, tf.int64),
        (
            {
                "input_ids": tf.TensorShape([None]),
                "attention_mask": tf.TensorShape([None]),
                "token_type_ids": tf.TensorShape([None]),
            },
            tf.TensorShape([]),
        ),
    )

# 对数据进行预处理
DATA_COLUMN = 'text'
LABEL_COLUMN = 'subreddit'

train_InputExamples = convert_data_to_examples(df_train, DATA_COLUMN, LABEL_COLUMN)

train_data = convert_examples_to_tf_dataset(list(train_InputExamples), tokenizer)
train_data = train_data.shuffle(100).batch(64).repeat(4)

# 创建早停回调
early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',  # 监控验证集的损失
    patience=1,  # 如果验证集的损失在3个epoch内没有改善，就停止训练
    restore_best_weights=True  # 恢复最佳权重
)

# 训练模型
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy('accuracy')])

model.fit(train_data, epochs=4, callbacks=[early_stopping_callback])  # 在fit函数中添加回调

# 假设你的模型叫做model
model.save_pretrained('my_model_directory')

# 读取测试数据集
df_test = pd.read_csv('dreaddit-test.csv')

# 填充缺失值
df_test['text'] = df_test['text'].fillna('')

# 将测试数据集中的'Assistance'，'Relationships'，'Homeless'，'Almosthomeless'这几个标签都替换为'noproblem'
df_test['subreddit'] = df_test['subreddit'].replace(['assistance', 'relationships', 'homeless', 'almosthomeless'], 'no problem')
df_test['subreddit'] = df_test['subreddit'].replace(['domesticviolence', 'survivorsofabuse'], 'may problem')

# # 删除'may problem'类的所有样本
# df_test = df_test[df_test['subreddit'] != 'may problem']

# 对文本数据进行预处理
df_test['text'] = df_test['text'].apply(preprocess_text)

# 使用之前训练好的LabelEncoder将标签转换为数字
df_test['subreddit'] = le.transform(df_test['subreddit'])

validation_InputExamples = convert_data_to_examples(df_test, DATA_COLUMN, LABEL_COLUMN)

validation_data = convert_examples_to_tf_dataset(list(validation_InputExamples), tokenizer)
validation_data = validation_data.batch(16)

# 使用之前训练好的模型进行预测
predictions_test = model.predict(validation_data)

# 如果测试数据集中包含真实的标签，可以计算准确率
accuracy_test = accuracy_score(df_test['subreddit'].values, predictions_test.logits.argmax(axis=1))
print(f'测试数据集的模型准确率: {accuracy_test:.3f}')
