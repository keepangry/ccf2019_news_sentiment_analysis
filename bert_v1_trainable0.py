# -*- coding: utf-8 -*-
# @Time    : 2019/8/26 上午7:37
# @Author  : yangsen
# @Email   : 0@keepangry.com
# @File    : bert_v1_trainable0.py
# @Software: PyCharm
from keras.utils import to_categorical

from config import BASE_PATH
from utils import TextClean, k_fold_split, f1
from custom_layers import base_multi_channel_softmax_net
import os
import keras
from collections import Counter
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from models import mr_base_model
import codecs
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf

gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

maxlen = 500
# learning_rate = 5e-5
# min_learning_rate = 1e-5


train_file = os.path.join(BASE_PATH, 'data/sentiment/Train_DataSet.csv')
test_file = os.path.join(BASE_PATH, 'data/sentiment/Test_DataSet.csv')
train_label_file = os.path.join(BASE_PATH, 'data/sentiment/Train_DataSet_Label.csv')

# 读取原始数据
origin_train_data = pd.read_csv(train_file, sep=',', encoding='utf-8')
origin_test_data = pd.read_csv(test_file, sep=',', encoding='utf-8')
origin_train_label_data = pd.read_csv(train_label_file, sep=',', encoding='utf-8')

# NaN给默认值
origin_train_data = origin_train_data.fillna("NN")
origin_test_data = origin_test_data.fillna("NN")


train_data_df = pd.merge(origin_train_data, origin_train_label_data, on=['id'])
train_data_df = train_data_df.dropna()

# 清洗文本
text_clean = TextClean()
train_data_df['content'] = train_data_df.apply(lambda x: '{}[SEP]{}'.format(x[1], x[2]), axis=1)
train_data_df['content'] = train_data_df['content'].apply(text_clean.clean)
del train_data_df['title']


test_data_df = origin_test_data.fillna("NN")
test_data_df['content'] = test_data_df.apply(lambda x: '{}[SEP]{}'.format(x[1], x[2]), axis=1)
test_data_df['content'] = test_data_df['content'].apply(text_clean.clean)
del test_data_df['title']

# 平均长度1188
# a = train_data_df['content'].apply(len)
print(np.mean(train_data_df['content'].apply(len).values))
print(np.mean(test_data_df['content'].apply(len).values))
#1150.0393732970026
#1157.482191408374

# for i in train_data_df['content'].values:
#     print(i)
#     input()



from keras_bert import load_trained_model_from_checkpoint, Tokenizer

config_path = '/home/yangsen/workspace/pretrain_data/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '/home/yangsen/workspace/pretrain_data/chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '/home/yangsen/workspace/pretrain_data/chinese_L-12_H-768_A-12/vocab.txt'

token_dict = {}

with codecs.open(dict_path, 'r', 'utf8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)


class OurTokenizer(Tokenizer):
    def _tokenize(self, text):
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            elif self._is_space(c):
                R.append('[unused1]') # space类用未经训练的[unused1]表示
            else:
                R.append('[UNK]') # 剩余的字符是[UNK]
        return R

tokenizer = OurTokenizer(token_dict)


def seq_padding(X, padding=0):
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])

class data_generator:
    def __init__(self, data, batch_size=32, is_shuffe=True):
        self.is_shuffe = is_shuffe
        self.data = data
        self.batch_size = batch_size
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1
    def __len__(self):
        return self.steps
    def __iter__(self):
        while True:
            idxs = list(range(len(self.data)))
            if self.is_shuffe:
                np.random.shuffle(idxs)
            X1, X2, Y = [], [], []
            for i in idxs:
                d = self.data[i]
                text = d[0][:maxlen]
                # 单输入
                #x1, x2 = tokenizer.encode(first=text)
                # 双输入
                title, content = text.split('[SEP]')
                x1, x2 = tokenizer.encode(first=title, second=content)

                y = d[1]
                X1.append(x1)
                X2.append(x2)
                Y.append(to_categorical(y, num_classes=3))
                if len(X1) == self.batch_size or i == idxs[-1]:
                    X1 = seq_padding(X1)
                    X2 = seq_padding(X2)
                    Y = seq_padding(Y)
                    yield [X1, X2], Y
                    [X1, X2, Y] = [], [], []


from keras.layers import *
from keras.models import Model
from keras.optimizers import Adam

bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, trainable=False, seq_len=None)
# import keras.backend as K
# K.set_floatx('float16')
# K.set_epsilon(1e-4)

# trainable 不可trainable 可以使用更长的序列
# for l in bert_model.layers:
#     l.trainable = False

x1_in = Input(shape=(None,))
x2_in = Input(shape=(None,))

x = bert_model([x1_in, x2_in])
x = Lambda(lambda x: x[:, 0])(x)
p = Dense(64, activation='relu')(x)
p = Dropout(0.2)(x)
p = Dense(3, activation='softmax')(p)


def macro_f1(y_true, y_pred, num_classes=3):
    def f1(y_true, y_pred):
        y_pred = K.cast(y_pred >= 0.5, 'float32')
        TP = K.sum(y_pred * y_true)
        precision = TP/(K.sum(y_pred)+0.0001)
        recall = TP/(K.sum(y_true)+0.0001)
        return 2*precision*recall / (precision+recall+0.0001)
    sum = 0
    for i in range(num_classes):
        sum += f1(y_true[..., i], y_pred[..., i])
    return K.cast(sum/num_classes, 'float32')


model = Model([x1_in, x2_in], p)
model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(0.001),  # 用足够小的学习率
    metrics=['accuracy', macro_f1]
)
model.summary()


train_data = train_data_df[['content', 'label']].values
np.random.seed(1)
np.random.shuffle(train_data)
train_data = [train_data[i] for i, j in enumerate(train_data) if i % 7 != 0]
valid_data = [train_data[i] for i, j in enumerate(train_data) if i % 7 == 0]


train_D = data_generator(train_data)
valid_D = data_generator(valid_data)

early_stop = keras.callbacks.EarlyStopping(monitor='acc', patience=3, restore_best_weights=True)

# Counter({2: 2931, 1: 3646, 0: 763})
# total: 7340,,   0:0.1  1:0.5  2:0.4
model.fit_generator(
    train_D.__iter__(),
    steps_per_epoch=len(train_D),
    # steps_per_epoch=10,
    epochs=20,
    validation_data=valid_D.__iter__(),
    validation_steps=len(valid_D),
    # validation_steps=2,
    class_weight={0: 10, 1: 5, 2: 4},
    callbacks=[early_stop]
)


is_test = False

if is_test:
    # 测试集
    fw = open('bert_trainable0_len500_segment12_0.745_20190826.csv', 'w')
    fw.write('id,label\n')

    test_data_df['label'] = test_data_df['content'].apply(lambda x: 0)
    test_data = test_data_df[['content', 'label']].values
    test_D = data_generator(test_data, batch_size=1, is_shuffe=False)
    t = test_D.__iter__()

    count = 0
    for i in range(len(test_data)):
        x = model.predict(next(t)[0])
        fw.write('{},{}\n'.format(test_data_df.values[i][0], np.argmax(x[0])))
        count += 1
        if count % 100 == 0:
            print(count)
    fw.close()
