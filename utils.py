# -*- coding: utf-8 -*-
# @Time    : 2019/8/23 下午7:46
# @Author  : yangsen
# @Email   : 0@keepangry.com
# @File    : utils.py
# @Software: PyCharm

import re
import numpy as np
from keras import backend as K


class TextClean(object):

    def __init__(self):
        self.repeat_pattern_dict = {
            r'\.{2,}': '..',
            r'。{2,}': '。。',
            r' {1,}': ' ',
            r'\\n': '',
            r'[\xa0]': '',
            r'[\\u3000|\\u200b|\u3000]': '',
            r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b': '',
            r'[a-zA-Z0-9\\\/\.\<\>\_\-\=\?\ \:\;\)\(\'\"]{8,}': '*',
            r'\s': '',
        }

    # 连续3个字符相同，压缩为2个
    def remove_pattern(self, content):
        # 去除多余空格
        for k, v in self.repeat_pattern_dict.items():
            content = re.sub(k, v, content)
        return content

    # 去除html
    def remove_html(self, content):
        pattern = re.compile(r'<[^>]+>', re.S)
        return pattern.sub('', content)

    def clean(self, content):
        return self.remove_pattern(self.remove_html(content))


class Dataset(object):
    def __init__(self, x_train, y_train, x_val=None, y_val=None, x_test=None, y_test=None):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.x_test = x_test
        self.y_test = y_test


def k_fold_split(x, y, k=5):
    assert x.shape[0] == y.shape[0]
    data_size = x.shape[0]
    fold_sample_num = data_size // k
    datasets = []
    for i in range(k):
        x_val = x[i * fold_sample_num: (i+1) * fold_sample_num]
        y_val = y[i * fold_sample_num: (i+1) * fold_sample_num]

        x_train = np.concatenate([
            x[: i * fold_sample_num],
            x[(i+1) * fold_sample_num:],
        ], axis=0)

        y_train = np.concatenate([
            y[: i * fold_sample_num],
            y[(i + 1) * fold_sample_num:],
        ], axis=0)
        datasets.append(Dataset(x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val))
    return datasets


def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.
        Only computes a batch-wise average of recall.
        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.
        Only computes a batch-wise average of precision.
        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

if __name__ == "__main__":
    tc = TextClean()
    print(type("子疑。饮醉竟..“濑尿”打开车。。。门趴在    驾驶室, ..............."))
    print("赌场罪。\\u200b北京中\\u3000剑律")
    print(tc.remove_pattern("正好住我们家附近。http://log.sina.com.cn/s/log_161eee1yac.htmlhttp://log.sina.com.cn/s/log_161eee1ya.htmlhttp://log.sina.com.cn/s/log_d8c8e1ya.html"))
    print(tc.remove_pattern("赌场罪。\\u200b北京中\\u3000剑律"))
    print(tc.remove_pattern("的问题。不属实无无\u3000\u3000陵城区郑家寨"))

    print(tc.remove_pattern("子疑。饮醉竟..“濑尿”打开车。。。门趴在    驾驶室, ..............."))
    print(tc.remove_html("子疑饮醉竟“濑尿”打开车门趴在驾驶室, ..............."))
    print(tc.clean("车 半\xa0年内骗\\n贷304万元,<!--enpcontent-->	 龙虎网讯 7名年轻人"))
    s = """
    这几天看了有人举报施某某的贴子，经与举报人联系证实，是宣某当天中午请举报人和枪手喝酒后，晚上才发的贴子！ 本人不去讨论前二天的举报，相信总归会有说法的！ 今天一看施全军2017年1月2日实名举报上黄镇宣国才的贴子（仍被锁定禁止评论）已经正好一整年了 =750) window.open(\'http://img.jsly001.com/attachment/mon_1801/4_291085_c796a6a86e17121.jpg?123\');" onload="if(this.offsetwidth>\'750\')this.width=\'750\';" src="http://img.jsly001.com/attachment/mon_1801/4_291085_c796a6a86e17121.jpg?123" style="max-width:750px;"/>图片:/home/alidata/www/data/tmp/qfupload/4_291085_1514981471478952.jpg 施全军实名举报50天后，上黄镇党委政府回复如下图： =750) window.open(\'http://img.jsly001.com/attachment/mon_1801/4_291085_a9b11b7ea2b1ce9.jpg?90\');" onload="if(this.offsetwidth>\'750\')this.width=\'750\';" src="http://img.jsly001.com/attachment/mon_1801/4_291085_a9b11b7ea2b1ce9.jpg?90" style="max-width:750px;"/>图片:/home/alidata/www/data/tmp/qfupload/4_291085_1514981472631668.jpg =750) window.open(\'http://img.jsly001.com/attachment/mon_1801/4_291085_9cde9b3943fe20c.jpg?75\');" onload="if(this.offsetwidth>\'750\')this.width=\'750\';" src="http://img.jsly001.com/attachment/mon_1801/4_291085_9cde9b3943fe20c.jpg?75" style="max-width:750px;"/>图片:/home/alidata/www/data/tmp/qfupload/4_291085_1514981472353075.jpg 一年的贴子，再次被网友顶起来后，才发现施某几天前回复网友的处理结果竟如下图： =750) window.open(\'http://img.jsly001.com/attachment/mon_1801/4_291085_9d32ee572760d85.jpg?131\');" onload="if(this.offsetwidth>\'750\')this.width=\'750\';" src="http://img.jsly001.com/attachment/mon_1801/4_291085_9d32ee572760d85.jpg?131" style="max-width:750px;"/>图片:/home/alidata/www/data/tmp/qfupload/4_291085_1514981473547172.jpg 现责问张涛书记： 1、宣国才被举报这么多问题，什么时候有答复。 2、宣国才被举报后，为什么被立刻免了村书记职务？为什么又被安排到城管队“吃空响”，自己却天天在我们水泥厂上班赚黑钱？ 3、这几个月，水泥每吨近200元纯利润，还供不应求，宣国才还清上黄政府担保借给宣国才代付振东厂工资社保的钱了吗？ 4、据了解宣国才占他人企业经营，又欠税52.16万元、欠社保32.76万元、应该还欠了职工工资几十万，上黄政府打算替宣国才担保还是归还？ 5、我们厂合法会计和老板被判刑四到六年，现在服刑。厂子给宣国才强占，宣国才每天赚20多万净利润，却对外宣称天天亏本！等咱老板刑满回厂，宣国才给咱厂“天天亏”可能要“亏”的几千万元，甚至几个亿，张涛书记您承担还是上黄政府承担？当初可是您亲自把厂交给宣国才生产的！ 希望徐市长看到本贴后能像批示263 、批示违建等民生问题一样，关注一下我们水泥厂的将来！也请徐市长抽日理万机之空亲自约谈一下当事人（特别是那位施站长），千万不能听取一面之辞！
    """
    # pattern = r'[a-zA-Z0-9\\\/\.\<\>\_\-\=\?\ \:\;\)\(\'\"]{8,}'
    # match = re.findall(pattern, s)
    # print(match)
    # print(s)
    # print(re.sub(pattern, ' ', s))
    print(tc.clean(s))

    s1 = " 刘晓） 					 来源： 南"
    print(re.sub('\s', '', s1))


    # print(match.groups())
    #print(tc.clean(s.strip()))

