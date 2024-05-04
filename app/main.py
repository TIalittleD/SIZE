from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import cv2
import time
from collections import Counter
import CNN_Model
import ImageCutting

app = Flask(__name__)

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

def cv2ImgAddText(img, text, left, top, textColor=(255, 0, 0), textSize=20):
    if (isinstance(img, np.ndarray)):  
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    fontStyle = ImageFont.truetype("Font\行者笔记简.ttf", textSize, encoding="utf-8")
    draw.text((left, top), text, textColor, font=fontStyle)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

@app.route('/process_image', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({'error': 'No selected image'}), 400

    img = cv2.imdecode(np.fromstring(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
    # 进行图片处理
    result_img = process_image_logic(img)

    # 将处理后的图片转换为Base64格式，方便返回给前端
    _, buffer = cv2.imencode('.jpg', result_img)
    result_img_base64 = base64.b64encode(buffer).decode('utf-8')

    return jsonify({'result_image': result_img_base64}), 200

def process_image_logic(img):
    all_mark_boxs, all_char_imgs, img_o = ImageCutting.divImgFromCV2(img)

    model = CNN_Model.create_custom_model((24, 24, 1), 15)
    model.load_weights('checkpoint/char_checkpoint.weights.h5')
    class_name = np.load('checkpoint/class_name.npy')

    for i in range(0, len(all_char_imgs)):
        row_imgs = all_char_imgs[i]
        for j in range(0, len(row_imgs)):
            block_imgs = row_imgs[j]
            block_imgs = np.array(block_imgs)
            results = CNN_Model.predict(model, block_imgs, class_name)
            print('recognize result:', results)
            result = calculation(results)
            print('calculate result:', result)
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
    return img_o

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0',port=int(os.environ.get('PORT', 80)))
