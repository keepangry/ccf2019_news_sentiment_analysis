# -*- coding: utf-8 -*-
# @Time    : 2019/8/23 下午7:37
# @Author  : yangsen
# @Email   : 0@keepangry.com
# @File    : v1.py
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

import pandas as pd
from sklearn.model_selection import train_test_split

train_file = os.path.join(BASE_PATH, 'data/sentiment/Train_DataSet.csv')
test_file = os.path.join(BASE_PATH, 'data/sentiment/Test_DataSet.csv')
train_label_file = os.path.join(BASE_PATH, 'data/sentiment/Train_DataSet_Label.csv')

# 读取原始数据
origin_train_data = pd.read_csv(train_file, sep=',')
origin_test_data = pd.read_csv(test_file, sep=',')
origin_train_label_data = pd.read_csv(train_label_file, sep=',')

# NaN给默认值
origin_train_data = origin_train_data.fillna("NN")
origin_test_data = origin_test_data.fillna("NN")


train_data = pd.merge(origin_train_data, origin_train_label_data, on=['id'])
train_data = train_data.dropna()

# 清洗文本
text_clean = TextClean()
train_data['content'] = train_data.apply(lambda x: '{}__SEP__{}'.format(x[1], x[2]), axis=1)
train_data['content'] = train_data['content'].apply(text_clean.clean)
del train_data['title']


test_data = origin_test_data.fillna("NN")
test_data['content'] = test_data.apply(lambda x: '{}__SEP__{}'.format(x[1], x[2]), axis=1)
test_data['content'] = test_data['content'].apply(text_clean.clean)
del test_data['title']


#test_data['content'].loc[2]

# 列左右合并
# pd.concat([origin_train_data, origin_train_label_data], axis=1)

# 查看空值行
# origin_train_label_data[pd.isna(origin_train_label_data['label'])]


# train_test_split
labels = train_data['label'].values


vocabulary_size = 5000

from keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer(num_words=vocabulary_size, char_level=True)
tokenizer.fit_on_texts(train_data['content'].values)
data = tokenizer.texts_to_sequences(train_data['content'].values)

maxlen = 200
data = pad_sequences(data, padding="pre", maxlen=maxlen)
labels = to_categorical(labels)

print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

# X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, random_state=1)

k = 5
datasets = k_fold_split(x=data, y=labels, k=k)

result = []
for i in range(k):
    dataset = datasets[i]
    print(dataset.x_train.shape)
    # exit()
    model = base_multi_channel_softmax_net(vocabulary_size, time_steps=maxlen)
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['acc', f1])
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            mode='auto'
        )
    ]
    #model.summary()
    history = model.fit(dataset.x_train, dataset.y_train,
                        epochs=20,
                        batch_size=128,
                        validation_data=(dataset.x_val, dataset.y_val),
                        callbacks=callbacks,
                        verbose=2)

    best_iter = np.argmax(history.history['val_acc'])
    print("fold #%s, best_iter: %s,  acc:%.4f" % (i, best_iter, history.history['val_acc'][best_iter]))
    result.append(history.history['val_acc'][best_iter])

print(np.mean(result))


## predict

test = tokenizer.texts_to_sequences(test_data['content'].values)
test = pad_sequences(test, padding="pre", maxlen=maxlen)

test_predict = model.predict(test)

fw = open('baseline_20190823.csv', 'w')
fw.write('id,label\n')
for i in range(test_data.shape[0]):
    fw.write('{},{}\n'.format(test_data.values[i][0], np.argmax(test_predict[i])))
fw.close()
