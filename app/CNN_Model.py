import os
import pathlib
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint


def create_custom_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Rescaling(1. / 255),
        layers.Conv2D(24, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(96, activation='relu'),
        layers.Dense(15)
    ])
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model


def prepare_dataset(data_dir, img_width, img_height, batch_size, validation_split=0.2, seed=123):
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=validation_split,
        subset='training',
        seed=seed,
        color_mode="grayscale",
        image_size=(img_height, img_width),
        batch_size=batch_size
    )
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=validation_split,
        subset="validation",
        seed=seed,
        color_mode="grayscale",
        image_size=(img_height, img_width),
        batch_size=batch_size
    )
    # 数据集预处理
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds


def train_model(model, train_ds, val_ds, epochs, checkpoint_path):
    # 创建一个保存模型权重的回调函数
    cp_callback = ModelCheckpoint(filepath=checkpoint_path,
                                  save_weights_only=True,
                                  save_best_only=True,
                                  verbose=1)
    # 训练模型
    model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=[cp_callback])


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


if __name__ == '__main__':
    data_dir = pathlib.Path('dataset')
    img_width, img_height = 24, 24
    batch_size = 64
    num_classes = 15
    input_shape = (img_height, img_width, 1)  # 使用灰度图像作为输入
    epochs = 10
    checkpoint_path = "checkpoint/char_checkpoint.weights.h5"
    train_ds, val_ds = prepare_dataset(data_dir, img_width, img_height, batch_size)
    model = create_custom_model(input_shape, num_classes)
    train_model(model, train_ds, val_ds, epochs, checkpoint_path)
