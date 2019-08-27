# -*- coding: utf-8 -*-
# @Time    : 2019/8/26 下午9:02
# @Author  : yangsen
# @Email   : 0@keepangry.com
# @File    : matrics.py
# @Software: PyCharm

import keras.backend as K
from sklearn.metrics import f1_score
import numpy as np

from keras.utils import to_categorical

y_true = np.array([0, 1, 1, 0, 1, 2])
y_pred = np.array([0, 2, 1, 0, 1, 1])
#
score = f1_score(y_true, y_pred, average='macro')
print(score)
#
# def pos_precision(y_true, y_pred):
#     # 预测为1
#     y_pred = K.cast(y_pred >= 0.9, 'float32')
#     P = K.sum(y_pred)
#     TP = K.sum(y_pred * y_true)
#     result = TP / (P+0.1)
#     return result
#
# print(K.get_session().run(pos_precision(y_true, y_pred)))
#


def macro_f1(y_true, y_pred):
    def f1(y_true, y_pred):
        y_pred = K.cast(y_pred >= 0.5, 'float32')
        TP = K.sum(y_pred * y_true)
        precision = TP/(K.sum(y_pred)+0.0001)
        recall = TP/(K.sum(y_true)+0.0001)
        if TP == 0:
            return 0.0
        return 2*precision*recall / (precision+recall+0.0001)
    num_classes = 3
    sum = 0
    for i in range(num_classes):
        sum += f1(y_true[..., i], y_pred[..., i])
    return K.cast(sum/num_classes, 'float32')


y_true = K.constant(to_categorical(y_true, num_classes=3))
y_pred = K.constant(to_categorical(y_pred, num_classes=3))
sess = K.get_session()
print(sess.run(y_true))
print(sess.run(y_pred))
score = sess.run(macro_f1(y_true, y_pred))
print(score)


if __name__ == "__main__":
    pass