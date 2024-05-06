from flask import Flask, request, jsonify,render_template
from PIL import Image, ImageDraw, ImageFont
import os
import base64
import numpy as np
import cv2
import CNN_Model
import ImageCutting

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
    return result, c_r


# 绘制文本
def cv2ImgAddText(img, text, left, top, textColor=(255, 0, 0), textSize=20):
    if (isinstance(img, np.ndarray)):
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    fontStyle = ImageFont.truetype("Font/行者笔记简.ttf", textSize, encoding="utf-8")
    draw.text((left, top), text, textColor, font=fontStyle)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

def process_image(img_bytes):
    img_np = np.frombuffer(img_bytes, dtype=np.uint8)
    img_cv2 = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

    all_mark_boxs, all_char_imgs, img_o = ImageCutting.divImg(img_cv2, False)
    model = CNN_Model.create_custom_model((24, 24, 1), 15)
    model.load_weights('checkpoint/char_checkpoint.weights.h5')
    class_name = np.load('checkpoint/class_name.npy')

    for i in range(0, len(all_char_imgs)):
        row_imgs = all_char_imgs[i]
        for j in range(0, len(row_imgs)):
            block_imgs = row_imgs[j]
            block_imgs = np.array(block_imgs)
            results = CNN_Model.predict(model, block_imgs, class_name)
            result, c_r = calculation(results)
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
            if str(result) != str(c_r):
                answer_text = f'({c_r})'
                img_o = cv2ImgAddText(img_o, answer_text, answer_box[2] + iw, answer_box[1], color, int(textSize * 0.5))

    ret, processed_img_bytes = cv2.imencode('.jpg', img_o)
    return processed_img_bytes.tobytes()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return jsonify({'error': 'No image sent'})

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # 读取上传的图片文件的字节流
    img_bytes = file.read()

    # 处理上传的图像
    processed_img_bytes = process_image(img_bytes)

    # 将处理后的图像以 Base64 编码的方式返回给前端
    processed_img_base64 = base64.b64encode(processed_img_bytes).decode('utf-8')

    return jsonify({
        'original_image': 'data:image/jpeg;base64,' + base64.b64encode(img_bytes).decode('utf-8'),
        'processed_image': 'data:image/jpeg;base64,' + processed_img_base64
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 80)))

