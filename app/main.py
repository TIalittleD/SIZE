# app.py

from flask import Flask, request, jsonify, render_template
import base64
import progressImage

app = Flask(__name__)

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
    processed_img_bytes = progressImage.process_image(img_bytes)

    # 将处理后的图像以 Base64 编码的方式返回给前端
    processed_img_base64 = base64.b64encode(processed_img_bytes).decode('utf-8')

    return jsonify({
        'original_image': 'data:image/jpeg;base64,' + base64.b64encode(img_bytes).decode('utf-8'),
        'processed_image': 'data:image/jpeg;base64,' + processed_img_base64
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=80)
