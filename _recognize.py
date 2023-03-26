import cv2 as cv
import numpy as np
import tensorflow as tf

digits = []
num_row = 1
# map:0mnist,1letter,2class
map = '0'
# model:0best,emnist_letter
modelchoice = '0'
mappings = {}
root = '.\src\plugins\\nonebot2_uocr_low'
model = tf.keras.models.load_model(root + r'\best.h5')

def setmodelchoice(modelchoice):
    global model
    if modelchoice == '0':
        model = tf.keras.models.load_model(root + r'\best.h5')
    elif modelchoice == '1':
        model = tf.keras.models.load_model(root + r'\emnist_leter.h5')
    elif modelchoice == '2':
        model = tf.keras.models.load_model(root + r'\emnist.h5')


def setmatchoice(map):
    global mappings
    if map == '1':
        # 读取emnist-letters-mapping.txt文件
        with open(root + r'\emnist-leters-mapping.txt') as f:
            lines = f.readlines()
        # 创建一个字典，将数字映射到字母
        mappings = {}
        for line in lines:
            index, upper, lower = line.split()  # 分割三个值
            label = chr(int(lower))  # 将大写字母的ASCII码转换为字符
            mappings[int(index)] = label  # 索引从1开始
    elif map == '2':
        with open(root + r"\emnist-byclass-mapping.txt") as f:
            lines = f.readlines()
        # 创建一个字典，将数字映射到字符
        mappings = {}
        for line in lines:
            index, label = line.split()  # 分割两个值
            label = chr(int(label))  # 将ASCII码转换为字符
            mappings[int(index)] = label  # 索引从0开始


def sort_contour(contours, height, num_row):
    contours = sorted(contours, key=lambda c: cv.minAreaRect(c)[0][1])
    # 根据轮廓中心y坐标将轮廓分成多行
    rows = []
    current_row = []
    current_y = None
    for contour in contours:
        center_x, center_y = cv.minAreaRect(contour)[0]
        if current_y is None:
            current_y = center_y
        if abs(center_y - current_y) > height / num_row * 0.4:  # 根据需要调整阈值
            rows.append(sorted(current_row, key=lambda c: cv.minAreaRect(c)[0][0]))
            current_row = []
            current_y = center_y
        current_row.append(contour)
    rows.append(sorted(current_row, key=lambda c: cv.minAreaRect(c)[0][0]))
    # 将所有轮廓连接成一个列表
    newcontours = [contour for row in rows for contour in row]
    return newcontours


def crop_image(img, num_row):
    # 将图像转换为numpy数组
    img = np.array(img)
    # cv.imwrite('aaa.png', img)
    # 计算图像的高度和宽度
    h_min, w_min = img.shape[:2]
    # 将图像转换为灰度图像
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # 对图像进行阈值处理进行二值化
    ret, thresh = cv.threshold(gray, 120, 255, cv.THRESH_BINARY_INV)
    # # 为形态学操作定义一个内核
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (10, 10))
    # 执行形态学闭合
    thresh = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)
    # 在图像中查找轮廓
    contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # 用书写习惯的排序算法排序轮廓
    contours = sort_contour(contours, h_min, num_row)
    # 创建一个列表来存储裁剪后的图像
    newimgs = []
    i = 0
    for none, cnt in enumerate(contours):
        # 获取外接矩形
        x, y, w, h = cv.boundingRect(cnt)
        # 绘制矩形框
        cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # 提取数字并保存为新的图片，如果满足条件
        digit = thresh[y:y + h, x:x + w]
        # 判断面积是否满足是数字的大小
        wd = digit.shape[1]
        hd = digit.shape[0]
        if hd > (h_min * 0.2 / num_row) or (num_row == 1 and hd > h_min * 0.1):
            # 保存图片
            if wd / hd > 2:
                continue
            height, width = digit.shape[:2]
            # 计算图像的最大边长
            max_side = max(height, width)
            # 创建一个正方形的白色背景
            square = np.full((max_side, max_side), 0, dtype=np.uint8)
            # 计算将图像放置在正方形中心的坐标
            x_pos = (max_side - width) // 2
            y_pos = (max_side - height) // 2
            # 将图像放置在正方形中心
            square[y_pos:y_pos + height, x_pos:x_pos + width] = digit
            # 将图像放置在正方形中心后，将其放置在一个黑色背景上
            img = cv.copyMakeBorder(square, 15, 15, 15, 15, cv.BORDER_CONSTANT, value=0)
            if height == 28 and width == 28:
                newimg = square
            else:
                # 将图像缩放为28x28像素
                newimg = cv.resize(img, (28, 28), interpolation=cv.INTER_NEAREST)
            # 将新图像添加到列表中
            newimgs.append(newimg)
    # 返回裁剪后的图像列表
    return newimgs


def predict_digit(img):
    if modelchoice == '0':
        img = img.reshape(1, 28, 28, 1)
        res = model.predict([img])[0]
        return np.argmax(res), max(res)
    else:
        img = tf.keras.preprocessing.image.img_to_array(img)
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        # 使用模型进行预测，得到一个概率分布
        pred = model.predict(img)
        # 找到概率最大的类别，并输出对应的标签

        class_index = np.argmax(pred)
        class_label = mappings[class_index]
        print(f"The predicted letter is {class_label} with probability {max(pred)}.")
        return class_label, max(max(pred))


def recognize(fullimg, num_row, mode):
    global modelchoice, map
    modelchoice = mode
    map = mode
    setmodelchoice(mode)
    setmatchoice(mode)
    cropped_images = crop_image(fullimg, num_row)
    digits = []
    predictions = []
    if len(cropped_images) == 0:
        result = '没有识别到字符'
        return result
    else:
        for img in cropped_images:
            digit, prediction = predict_digit(img)
            digits.append(digit)
            predictions.append(prediction)
        avgpredit = np.mean(predictions)
        strdigits = [str(x) for x in digits]
        strdigits = ''.join(strdigits)
        result = f'识别到的字符是:{strdigits},可信度为{avgpredit}'
        digits.clear()
        predictions.clear()
        return result
