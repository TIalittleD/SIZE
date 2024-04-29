import numpy as np
import cv2
import os
import shutil
import matplotlib.pyplot as plt

# 一行图片的X轴投影
def img_x_shadow(img_b):
    ### 计算投影 ###
    (h, w) = img_b.shape
    # 初始化一个跟图像宽一样长度的数组，用于记录每一列的像素数量
    a = [0 for z in range(0, w)]
    # 遍历每一列，记录下这一列包含多少有效像素点
    for i in range(0, w):
        for j in range(0, h):
            if img_b[j, i] == 255:
                a[i] += 1
    return a

# 整幅图片的Y轴投影
def img_y_shadow(img_b):
    ### 计算投影 ###
    (h, w) = img_b.shape
    # 初始化一个跟图像高一样长度的数组，用于记录每一行的黑点个数
    a = [0 for z in range(0, h)]
    # 遍历每一行，记录下这一行包含多少有效像素点
    for i in range(0, h):
        for j in range(0, w):
            if img_b[i, j] == 255:
                a[i] += 1
    return a

# 图片获取文字块，传入投影列表，返回标记的数组区域坐标[[左，上，右，下]]，用于切行
def img2rows(a, w, h):
    #根据投影切分图块
    inLine = False  # 是否已经开始切分
    start = 0  # 某次切分的起始索引
    mark_boxs = []
    for i in range(0, len(a)):
        if inLine == False and a[i] > 10:
            inLine = True
            start = i
        # 记录这次选中的区域[左，上，右，下]，上下就是图片的上下，左右是start到当前
        elif i - start > 5 and a[i] < 10 and inLine:
            inLine = False
            if i - start > 10:
                top = max(start - 1, 0)
                bottom = min(h, i + 1)
                box = [0, top, w, bottom]
                mark_boxs.append(box)

    return mark_boxs

# 图片获取文字块，传入图片路径，返回标记的数组区域坐标[[左，上，右，下]]，用于从行切块
def row2blocks(a, w, h):
    #根据投影切分图块
    inLine = False  # 是否已经开始切分
    start = 0  # 某次切分的起始索引
    block_mark_boxs = []  # 切分的矩形区域坐标[左，上，右，下]

    for i in range(0, len(a)):
        # 如果还没有开始切，并且这列有效像素超过2个
        if inLine == False and a[i] > 2:
            inLine = True  # 标记为现在开始切块
            start = i  # 标记这次切块的位置索引
        # 如果在切，并且已经超过10个，并且这次低于2个有效像素，说明遇到空白了
        elif i - start > 10 and a[i] < 2 and inLine:
            inLine = False  # 标记不切了
            # 记录这次选中的区域[左，上，右，下]，上下就是图片的上下，左右是start到当前
            left = max(start - 1, 0)
            right = min(w, i + 1)
            box = [left, 0, right, h]
            block_mark_boxs.append(box)
    return block_mark_boxs

# 图片获取文字块，传入图片路径，返回标记的数组区域坐标[[左，上，右，下]]，用于从块切字
def block2chars(a, w, h, row_top, block_left):
    #根据投影切分图块
    inLine = False  # 是否已经开始切分
    start = 0  # 某次切分的起始索引
    char_mark_boxs = []  # 切分的矩形区域坐标[左，上，右，下]
    abs_char_mark_boxs = []  # 切分的矩形区域坐标[左，上，右，下]

    for i in range(0, len(a)):
        # 如果还没有开始切，并且这列有效像素超过1个
        if inLine == False and a[i] > 0:
            inLine = True  # 标记为现在开始切块
            start = i  # 标记这次切块的位置索引
        # 如果在切，并且已经超过5个，并且这次低于2个有效像素，说明遇到空白了
        elif i - start > 5 and a[i] < 1 and inLine:
            inLine = False  # 标记不切了
            # 记录这次选中的区域[左，上，右，下]，上下就是图片，左右是start到当前
            left = max(start - 1, 0)
            right = min(w, i + 1)
            box = [left, 0, right, h]
            char_mark_boxs.append(box)
            ads_box = [block_left + left, row_top, block_left + right, row_top + h]
            abs_char_mark_boxs.append(ads_box)
    return char_mark_boxs, abs_char_mark_boxs

# 转化为方形图片
def get_square_img(image):
    x, y, w, h = cv2.boundingRect(image)
    image = image[y:y + h, x:x + w]

    max_size = 18
    max_size_and_border = 24

    if w > max_size or h > max_size:  # 有超过宽高的情况
        if w >= h:  # 宽比高长，压缩宽
            times = max_size / w
            w = max_size
            h = int(h * times)
        else:  # 高比宽长，压缩高
            times = max_size / h
            h = max_size
            w = int(w * times)
        # 保存图片大小
        image = cv2.resize(image, (w, h))

    xw = image.shape[0]
    xh = image.shape[1]

    xwLeftNum = int((max_size_and_border - xw) / 2)
    xwRightNum = (max_size_and_border - xw) - xwLeftNum

    xhLeftNum = int((max_size_and_border - xh) / 2)
    xhRightNum = (max_size_and_border - xh) - xhLeftNum

    img_large = np.pad(image,
                       ((xwLeftNum, xwRightNum),
                        (xhLeftNum, xhRightNum)),
                       'constant',
                       constant_values=(0, 0))

    return img_large

# 裁剪图片
def cut_img(img, mark_boxs, is_square=False):
    img_items = []
    for i in range(0, len(mark_boxs)):
        img_org = img.copy()
        box = mark_boxs[i]
        img_item = img_org[box[1]:box[3], box[0]:box[2]]

        if is_square:  # 是否转化为方形
            img_item = get_square_img(img_item)
        img_items.append(img_item)
    return img_items

# 保存图片
def save_imgs(dir_name, imgs):
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    img_paths = []
    for i in range(0, len(imgs)):
        file_path = dir_name + '/part_' + str(i) + '.png'
        cv2.imwrite(file_path, imgs[i])
        img_paths.append(file_path)

    return img_paths

# 图像切割，获取块的轮廓
def divImg(img_path, save_file=False):
    thresh = 200

    # 读入图片（色彩），仅用于最后输出
    img_o = cv2.imread(img_path, 1)

    # 读入图片（灰度），用于切块识别等
    img = cv2.imread(img_path, 0)

    # 锐化图像（灰度）
    img = cv2.filter2D(img, -1, kernel=np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]))
    (img_h, img_w) = img.shape

    # 二值化整个图，用于分行
    ret, img_b = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY_INV)

    # 计算投影，并截取整个图片的行
    img_y_shadow_a = img_y_shadow(img_b)

    row_mark_boxs = img2rows(img_y_shadow_a, img_w, img_h)
    # 切行的图片，切的是灰图原图
    row_imgs = cut_img(img, row_mark_boxs)
    all_mark_boxs = []
    all_char_imgs = []
    # 从行切块
    for i in range(0, len(row_imgs)):
        row_img = row_imgs[i]
        (row_img_h, row_img_w) = row_img.shape
        # 二值化一行的图，用于切块
        ret, row_img_b = cv2.threshold(row_img, thresh, 255, cv2.THRESH_BINARY_INV)
        kernel = np.ones((3, 3), np.uint8)
        # 图像膨胀6次
        row_img_b_d = cv2.dilate(row_img_b, kernel, iterations=6)
        img_x_shadow_a = img_x_shadow(row_img_b_d)

        block_mark_boxs = row2blocks(img_x_shadow_a, row_img_w, row_img_h)
        row_char_boxs = []
        row_char_imgs = []
        # 切块的图，切的是灰图原图
        block_imgs = cut_img(row_img, block_mark_boxs)
        if save_file:
            # 如果要保存切图
            b_imgs = save_imgs('imgs/cuts/row_' + str(i), block_imgs)
        # 从块切字
        for j in range(0, len(block_imgs)):
            block_img = block_imgs[j]
            (block_img_h, block_img_w) = block_img.shape
            # 二值化块,切字符图片
            ret, block_img_b = cv2.threshold(block_img, thresh, 255, cv2.THRESH_BINARY_INV)
            block_img_x_shadow_a = img_x_shadow(block_img_b)
            row_top = row_mark_boxs[i][1]
            block_left = block_mark_boxs[j][0]
            char_mark_boxs, abs_char_mark_boxs = block2chars(block_img_x_shadow_a,
                                                             block_img_w,
                                                             block_img_h,
                                                             row_top,
                                                             block_left)
            row_char_boxs.append(abs_char_mark_boxs)
            # 切的是二值化的图
            char_imgs = cut_img(block_img_b, char_mark_boxs, True)
            row_char_imgs.append(char_imgs)
            if save_file:
                # 如果要保存切图
                c_imgs = save_imgs('imgs/cuts/row_' + str(i) + '/blocks_' + str(j), char_imgs)
        all_mark_boxs.append(row_char_boxs)
        all_char_imgs.append(row_char_imgs)

    return all_mark_boxs, all_char_imgs, img_o
