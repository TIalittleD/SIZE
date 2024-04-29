from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import cv2
import time
import CNN_Model
import ImageCutting

app = Flask(__name__)

# 计算数值并返回结果
def calculation(chars):
    cstr = ''.join(chars)
    c_r = ''
    result = ''
    if ("=" in cstr):  # 有等号
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
            # if str(c_r) == str(r_str):
            #     result = c_r
            # else:
            #     result = c_r
    return result

# 绘制文本
def cv2ImgAddText(img, text, left, top, textColor=(255, 0, 0), textSize=20):
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype("Font\行者笔记简.ttf", textSize, encoding="utf-8")
    # 绘制文本
    draw.text((left, top), text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

# 主函数，执行图片处理
def main(path, save=False):
    # 获取切图标注，切图图片，原图图图片
    all_mark_boxs, all_char_imgs, img_o = ImageCutting.divImg(path, save)

    # 恢复模型，用于图片识别
    model = CNN_Model.create_custom_model((24, 24, 1), 15)
    model.load_weights('checkpoint/char_checkpoint.weights.h5')
    class_name = np.load('checkpoint/class_name.npy')

    # 遍历行
    for i in range(0, len(all_char_imgs)):
        row_imgs = all_char_imgs[i]
        # 遍历块
        for j in range(0, len(row_imgs)):
            block_imgs = row_imgs[j]
            block_imgs = np.array(block_imgs)
            # 图片识别
            results = CNN_Model.predict(model, block_imgs, class_name)
            print('recognize result:', results)
            # 计算结果
            result = calculation(results)
            print('calculate result:', result)
            # 获取块的标注坐标
            block_mark = all_mark_boxs[i][j]
            # 获取结果的坐标，写在块的最后一个字
            answer_box = block_mark[-1]
            # 计算最后一个字的位置
            x = answer_box[2]
            y = answer_box[3]
            iw = answer_box[2] - answer_box[0]
            ih = answer_box[3] - answer_box[1]
            # 计算字体大小
            textSize = max(iw, ih)
            # 根据结果设置字体颜色
            if str(result) == "√":
                color = (0, 255, 0)
            elif str(result) == "×":
                color = (255, 0, 0)
            else:
                color = (192, 192, 192)
            # 将结果写到原图上
            img_o = cv2ImgAddText(img_o, str(result), answer_box[2], answer_box[1], color, textSize)
    # 将写满结果的原图保存
    f_name = path[:path.index(".")] + "_Result" + path[path.index("."):]
    cv2.imwrite(f_name, img_o)
	
@app.route('/process_image', methods=['POST'])
def process_image():
    try:
        # 获取上传的图像文件
        file = request.files['image']
        
        # 保存图像文件到临时文件夹
        img_path = 'temp_image.png'
        file.save(img_path)
        
        # 调用处理函数处理图像
        result_image = main(img_path)
        
        # 返回处理后的图像文件
        return send_file(result_image, mimetype='image/png')
    except Exception as e:
        return jsonify({'code': 500, 'msg': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

