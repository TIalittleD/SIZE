from flask import Flask, render_template, request
from PIL import Image, ImageDraw, ImageFont  # 用于图像处理和绘图
import numpy as np
import cv2
import matplotlib.pyplot as plt  # 用于图像展示
import time  # 用于计时
from collections import Counter  # 用于计数
import CNN_Model  # 自定义的模型，用于字符识别
import ImageCutting  # 用于图像切割

app = Flask(__name__)

# 计算数值并返回结果
def calculation(chars):
    cstr = ''.join(chars)
    c_r = ''
    result = ''
    if ("=" in cstr):  
        str_arr = cstr.split('=')
        c_str = str_arr[0]
        r_str = str_arr[1]
        c_str = c_str.replace("×", "*")
        c_str = c_str.replace("÷", "/")
        try:
            c_r = int(eval(c_str))
        except Exception as e:
            print("Exception", e)
        if r_str == "":
            result = c_r
        else:
            if str(c_r) == str(r_str):
                result = "√"
            else:
                result = "×"
    return result

# 主函数，执行图片处理
def main(path, save=False):
    all_mark_boxs, all_char_imgs, img_o = ImageCutting.divImg(path, save)

    model = CNN_Model.create_custom_model((24, 24, 1), 15)
    model.load_weights('checkpoint/char_checkpoint.weights.h5')
    class_name = np.load('checkpoint/class_name.npy')

    for i in range(0, len(all_char_imgs)):
        row_imgs = all_char_imgs[i]
        for j in range(0, len(row_imgs)):
            block_imgs = row_imgs[j]
            block_imgs = np.array(block_imgs)
            results = CNN_Model.predict(model, block_imgs, class_name)
            result = calculation(results)
            block_mark = all_mark_boxs[i][j]
            answer_box = block_mark[-1]
            x = answer_box[2]
            y = answer_box[3]
            iw = answer_box[2] - answer_box[0]
            ih = answer_box[3] - answer_box[1]
            textSize = max(iw, ih)
            if str(result) == "√":
                color = (0, 255, 0)
            elif str(result) == "×":
                color = (255, 0, 0)
            else:
                color = (192, 192, 192)
            img_o = cv2ImgAddText(img_o, str(result), answer_box[2], answer_box[1], color, textSize)
    f_name = path[:path.index(".")] + "_Result" + path[path.index("."):]
    cv2.imwrite(f_name, img_o)
    return f_name

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template('index.html', message='未上传图片！')
        image = request.files['image']
        if image.filename == '':
            return render_template('index.html', message='未选择图片！')
        if image:
            image.save('uploaded_image.png')
            result_image_path = main('uploaded_image.png', save=True)
            return render_template('index.html', message='处理完成！', result_image=result_image_path)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
