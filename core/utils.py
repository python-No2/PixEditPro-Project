import numpy as np
import cv2 as cv


# 定义添加椒盐噪声函数
def AddSaltPepperNoise(src, rate):
    srcCopy = src.copy()
    height, width = srcCopy.shape[0:2]
    noiseCount = int(rate*height*width/2)
    # add salt noise
    X = np.random.randint(width, size=(noiseCount,))
    Y = np.random.randint(height, size=(noiseCount,))
    srcCopy[Y, X] = 255
    # add black peper noise
    X = np.random.randint(width, size=(noiseCount,))
    Y = np.random.randint(height, size=(noiseCount,))
    srcCopy[Y, X] = 0
    return srcCopy


# 风格转换-糖果
def trans2candy(image):

    # 加载模型
    net = cv.dnn.readNetFromTorch('.\models\candy.t7') # 选择一个模型的地址
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV) # 创建后端

    (h, w) = image.shape[:2]
    blob = cv.dnn.blobFromImage(image, 1, (w, h), (103.939, 116.779, 123.680), swapRB=False, crop=False)
    # 进行计算
    net.setInput(blob)
    out = net.forward()
    out = out.reshape(3, out.shape[2], out.shape[3])
    out[0] += 103.939
    out[1] += 116.779
    out[2] += 123.68
    out /= 255
    out = out.transpose(1, 2, 0)
    out = out * 255
    return out

# 风格转换-星空
def trans2star(image):
    # 加载模型
    net = cv.dnn.readNetFromTorch('.\models\starry_night.t7') # 选择一个模型的地址
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV) # 创建后端

    (h, w) = image.shape[:2]
    blob = cv.dnn.blobFromImage(image, 1, (w, h), (103.939, 116.779, 123.680), swapRB=False, crop=False)
    # 进行计算
    net.setInput(blob)
    out = net.forward()
    out = out.reshape(3, out.shape[2], out.shape[3])
    out[0]+= 103.939
    out[1]+= 116.779
    out[2]+= 123.68
    out /= 255
    out = out.transpose(1, 2, 0)
    out = out * 255
    return out

#风格转换-毕加索
def trans2bjs(image):

    # 加载模型
    net = cv.dnn.readNetFromTorch('.\models\composition_vii.t7') # 选择一个模型的地址
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV) # 创建后端

    (h, w) = image.shape[:2]
    blob = cv.dnn.blobFromImage(image, 1, (w, h), (103.939, 116.779, 123.680), swapRB=False, crop=False)
    # 进行计算
    net.setInput(blob)
    out = net.forward()
    out = out.reshape(3, out.shape[2], out.shape[3])
    out[0]+= 103.939
    out[1]+= 116.779
    out[2]+= 123.68
    out /= 255
    out = out.transpose(1, 2, 0)
    out = out * 255
    return out

# 风格转换-缪斯
def trans2ms(image):
    # 加载模型
    net = cv.dnn.readNetFromTorch('.\models\la_muse.t7')#选择一个模型的地址
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)#创建后端

    (h, w) = image.shape[:2]
    blob = cv.dnn.blobFromImage(image, 1, (w, h), (103.939, 116.779, 123.680), swapRB=False, crop=False)
    # 进行计算
    net.setInput(blob)
    out = net.forward()
    out = out.reshape(3, out.shape[2], out.shape[3])
    out[0]+= 103.939
    out[1]+= 116.779
    out[2]+= 123.68
    out /= 255
    out = out.transpose(1, 2, 0)
    out = out * 255
    return out

# 风格迁移-马赛克
def trans2mosaic(image):

    # 加载模型
    net = cv.dnn.readNetFromTorch('.\models\mosaic.t7') # 选择一个模型的地址
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV) # 创建后端

    (h, w) = image.shape[:2]
    blob = cv.dnn.blobFromImage(image, 1, (w, h), (103.939, 116.779, 123.680), swapRB=False, crop=False)
    # 进行计算
    net.setInput(blob)
    out = net.forward()
    out = out.reshape(3, out.shape[2], out.shape[3])
    out[0]+= 103.939
    out[1]+= 116.779
    out[2]+= 123.68
    out /= 255
    out = out.transpose(1, 2, 0)
    out = out * 255
    return out

# 风格迁移-神奈川冲浪里
def trans2snc(image):
    # 加载模型
    net = cv.dnn.readNetFromTorch('.\models\\the_wave.t7') # 选择一个模型的地址
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV) # 创建后端

    (h, w) = image.shape[:2]
    blob = cv.dnn.blobFromImage(image, 1, (w, h), (103.939, 116.779, 123.680), swapRB=False, crop=False)
    # 进行计算
    net.setInput(blob)
    out = net.forward()
    out = out.reshape(3, out.shape[2], out.shape[3])
    out[0]+= 103.939
    out[1]+= 116.779
    out[2]+= 123.68
    out /= 255
    out = out.transpose(1, 2, 0)
    out = out * 255
    return out

# 风格迁移-达达主义
def trans2dd(image):
    # 加载模型
    net = cv.dnn.readNetFromTorch(r'.\models\udnie.t7') # 选择一个模型的地址
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV) # 创建后端

    (h, w) = image.shape[:2]
    blob = cv.dnn.blobFromImage(image, 1, (w, h), (103.939, 116.779, 123.680), swapRB=False, crop=False)
    # 进行计算
    net.setInput(blob)
    out = net.forward()
    out = out.reshape(3, out.shape[2], out.shape[3])
    out[0]+= 103.939
    out[1]+= 116.779
    out[2]+= 123.68
    out /= 255
    out = out.transpose(1, 2, 0)
    out = out * 255
    return out

# 风格迁移-呐喊
def trans2nh(image):
    # 加载模型
    net = cv.dnn.readNetFromTorch('.\models\\the_scream.t7') # 选择一个模型的地址
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV) # 创建后端

    (h, w) = image.shape[:2]
    blob = cv.dnn.blobFromImage(image, 1, (w, h), (103.939, 116.779, 123.680), swapRB=False, crop=False)
    # 进行计算
    net.setInput(blob)
    out = net.forward()
    out = out.reshape(3, out.shape[2], out.shape[3])
    out[0]+= 103.939
    out[1]+= 116.779
    out[2]+= 123.68
    out /= 255
    out = out.transpose(1, 2, 0)
    out = out * 255
    return out

# 风格迁移-羽毛
def trans2fea(image):
    # 加载模型
    net = cv.dnn.readNetFromTorch('.\models\\feathers.t7') # 选择一个模型的地址
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV) # 创建后端

    (h, w) = image.shape[:2]
    blob = cv.dnn.blobFromImage(image, 1, (w, h), (103.939, 116.779, 123.680), swapRB=False, crop=False)
    # 进行计算
    net.setInput(blob)
    out = net.forward()
    out = out.reshape(3, out.shape[2], out.shape[3])
    out[0]+= 103.939
    out[1]+= 116.779
    out[2]+= 123.68
    out /= 255
    out = out.transpose(1, 2, 0)
    out = out * 255
    return out

# 图像修复
import cv2 as cv
import numpy as np

import cv2 as cv
import numpy as np

def colorizer(image):
    # 设置模型路径
    prototxt = r'models/colorization_deploy_v2.prototxt'
    model = r'models/colorization_release_v2.caffemodel'
    points = r'models/pts_in_hull.npy'
    net = cv.dnn.readNetFromCaffe(prototxt, model)  # 加载模型
    pts = np.load(points)

    # 添加聚类中心作为1x1卷积到模型中
    class8 = net.getLayerId("class8_ab")  # 获取class8_ab层的ID
    conv8 = net.getLayerId("conv8_313_rh")  # 获取conv8_313_rh层的ID
    pts = pts.transpose().reshape(2, 313, 1, 1)  # 转换并调整聚类中心点的形状
    net.getLayer(class8).blobs = [pts.astype("float32")]  # 设置class8_ab层的权重
    net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]  # 设置conv8_313_rh层的权重

    # 读取图像并转换颜色空间
    scaled = image.astype("float32") / 255.0  # 将图像像素值缩放到[0,1]范围
    lab = cv.cvtColor(scaled, cv.COLOR_BGR2LAB)  # 将图像从BGR转换到LAB颜色空间

    # 调整图像大小并分离L通道
    resized = cv.resize(lab, (224, 224))  # 将图像调整到224x224大小
    L = cv.split(resized)[0]  # 分离L通道
    L -= 50  # 中心化L通道

    # 预测a和b通道
    net.setInput(cv.dnn.blobFromImage(L))  # 将L通道作为输入
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))  # 预测ab通道并调整形状

    # 调整ab通道大小与原图匹配
    ab = cv.resize(ab, (image.shape[1], image.shape[0]))  # 调整ab通道大小到原图大小

    # 合并L通道和预测的ab通道
    L = cv.split(lab)[0]  # 分离原图的L通道
    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)  # 合并L和ab通道

    # 转换回BGR颜色空间并裁剪无效值
    colorized = cv.cvtColor(colorized, cv.COLOR_LAB2BGR)  # 将图像从LAB转换回BGR
    colorized = np.clip(colorized, 0, 1)  # 裁剪像素值到[0,1]范围

    # 转换为8位无符号整数
    colorized = (255 * colorized).astype("uint8")  # 将像素值转换到[0,255]范围并转换为uint8类型

    # 应用双边滤波以减少噪声同时保留边缘
    colorized = cv.bilateralFilter(colorized, d=3, sigmaColor=30, sigmaSpace=30)

    # 应用CLAHE进行对比度增强
    lab = cv.cvtColor(colorized, cv.COLOR_BGR2LAB)
    l, a, b = cv.split(lab)
    clahe = cv.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    lab = cv.merge((cl, a, b))
    colorized = cv.cvtColor(lab, cv.COLOR_LAB2BGR)

    # 应用锐化滤波器以增强图像清晰度
    kernel = np.array([[0, -0.5, 0],
                       [-0.5, 3,-0.5],
                       [0, -0.5, 0]])
    colorized = cv.filter2D(colorized, -1, kernel)

    return colorized


