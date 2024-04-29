import os
import pathlib
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint


#预测
def predict(model, imgs, class_name):
    label_dict = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4',
                  5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
                  10: '=', 11: '+', 12: '-', 13: '×', 14: '÷'}
    # 预测图片，获取预测值
    predicts = model.predict(imgs)
    results = []  # 保存结果的数组
    for predict in predicts:  # 遍历每一个预测结果
        index = np.argmax(predict)  # 寻找最大值
        result = class_name[index]  # 取出字符
        results.append(label_dict[int(result)])
    return results
