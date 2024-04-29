from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os

# 导入图像处理相关的函数
from PIL import Image
import numpy as np
import cv2
from CalculationAndDraw import main  # 替换成你的图像处理模块

app = Flask(__name__)

# 设置上传文件存储目录
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 允许上传的文件类型
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# 检查文件类型是否允许
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# 首页，上传图片的界面
@app.route('/')
def index():
    return render_template('index.html')

# 图片处理接口
@app.route('/process_image', methods=['POST'])
def process_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        # 调用图像处理函数
        main(filepath, save=True)  # 保存处理后的图片
        # 返回处理后的图片路径
        processed_filename = 'processed_' + filename
        processed_filepath = os.path.join(app.config['UPLOAD_FOLDER'], processed_filename)
        return jsonify({'processed_image': processed_filepath})
    else:
        return jsonify({'error': 'File type not allowed'})

if __name__ == '__main__':
    app.run(debug=True)
