import os
import cv2 as cv
import numpy as np
import core.utils as ut

# 转到对应操作中处理图像
def pre_process(data_path, num, ext):
    file_name = os.path.split(data_path)[1].split('.')[0]
    img = cv.imdecode(np.fromfile(data_path, dtype=np.uint8), -1)

    if num == 0:  # 椒盐噪声
        img = AddSaltPepperNoise(img, 0.5)

    elif num == 1:  # 均值平滑
        img = cv.blur(img, (3, 3))
    elif num == 2:  # 中值平滑
        img = cv.medianBlur(img, 3)
    elif num == 3:  # 高斯平滑
        img = cv.GaussianBlur(img,(11,11),0)

    elif num == 4: # 图像锐化-拉普拉斯算子
        gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        img=cv.Laplacian(gray,cv.CV_16S,ksize=3)
        img=cv.convertScaleAbs(img)
    elif num==5: # 图像锐化-Sobel算子水平方向
        gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        img=cv.Sobel(gray,cv.CV_64F,1,0,ksize=5)
    elif num==6: # 图像锐化-Sobel算子垂直方向
        gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        img=cv.Sobel(img,cv.CV_64F,0,1,ksize=5)
    elif num==7: # 将图像用双线性插值法扩大图像
        img = cv.resize(img, (0, 0), fx=2, fy=2, interpolation=cv.INTER_NEAREST)
    elif num==8: # 左移30个像素，下移50个像素
        M=np.float32([[1,0,30],[0,1,60]])
        height, width, channel = img.shape
        img = cv.warpAffine(img, M, (width, height))
    elif num==9: # 旋转45度，缩放因子为1
        height, width, channel = img.shape
        M = cv.getRotationMatrix2D(((width/2),(height/2)),45,1)
        tmp=cv.warpAffine(img, M, (2*width, 2*height))
        img=tmp

    elif num==10: # 转灰度图
        img= cv.cvtColor(img,cv.COLOR_BGR2GRAY)

    elif num==11: # 转灰度后二值化-全局阈值法
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret, img = cv.threshold(img, 127, 255, cv.THRESH_BINARY)

    elif num==12: # 直方图均衡化
        img= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        src = cv.resize(img, (256,256))
        img=cv.equalizeHist(src)
    elif num==13: # 灰度直方图
        img=cv.calcHist([img],[0],None,[256],[0,255])
    elif num==14: # 仿射变换
        src = cv.resize(img, (256, 256))
        rows, cols = src.shape[: 2]
        post1=np.float32([[50,50],[200,50],[50,200]])
        post2=np.float32([[10,100],[200,50],[100,250]])
        M=cv.getAffineTransform(post1,post2)
        img=cv.warpAffine(src,M,(rows,cols))
    elif num==15: #透视变换
        src = cv.resize(img, (256, 256))
        rows, cols = src.shape[: 2]
        post1 = np.float32([[55, 65], [288, 49], [28, 237], [239,240]])
        post2 = np.float32([[0, 0], [200, 0], [0, 200], [200, 200]])
        M = cv.getPerspectiveTransform(post1, post2)
        img=cv.warpPerspective(src,M,(200,200))
    elif num==16: # 图像翻转
        gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        img=255-img
    elif num==17: #rgb转hsv
        img=cv.cvtColor(img,cv.COLOR_RGB2HSV)
    elif num==18: #hsv获取h
        hsv = cv.cvtColor(img, cv.COLOR_RGB2HSV)
        img = hsv[:,:,0]
    elif num==19: #hsv获取s
        hsv = cv.cvtColor(img, cv.COLOR_RGB2HSV)
        img = hsv[:,:,1]
    elif num==20: #hsv获取v
        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        img = hsv[:,:,2]
    elif num==21: #rgb获取b
        img=img[:,:,0]
    elif num==22: #rgb获取g
        img=img[:,:,1]
    elif num==23: #rgb获取r
        img=img[:,:,2]
    elif num==24: #水平翻转
        img=cv.flip(img,1,dst=None)
    elif num==25: #垂直翻转
        img=cv.flip(img,0,dst=None)
    elif num==26: #对角镜像
        img=cv.flip(img,-1,dst=None)
    elif num==27: #图像开运算
        kernel=cv.getStructuringElement(cv.MORPH_CROSS,(5,5))
        img=cv.morphologyEx(img,cv.MORPH_OPEN,kernel)
    elif num==28: #图像闭运算
        kernel=cv.getStructuringElement(cv.MORPH_CROSS,(5,5))
        img=cv.morphologyEx(img,cv.MORPH_CLOSE,kernel)
    elif num==29: #腐蚀
        kernel=cv.getStructuringElement(cv.MORPH_CROSS,(5,5))
        img=cv.erode(img,kernel)
    elif num==30: #膨胀
        kernel=cv.getStructuringElement(cv.MORPH_CROSS,(5,5))
        img=cv.dilate(img,kernel)
    elif num==31: #顶帽运算
        kernel=cv.getStructuringElement(cv.MORPH_CROSS,(5,5))
        img=cv.morphologyEx(img,cv.MORPH_TOPHAT,kernel)
    elif num == 32:  # 底帽运算
        kernel = cv.getStructuringElement(cv.MORPH_CROSS,(5,5))
        img = cv.morphologyEx(img, cv.MORPH_TOPHAT, kernel)
    elif num == 33:  # houghLinesP实现线条检测
        img = cv.GaussianBlur(img, (3, 3), 0)
        edges = cv.Canny(img, 50, 150, apertureSize=3)
        minLineLength = 200
        maxLineGap = 15
        linesP = cv.HoughLinesP(edges,1,np.pi/180,80,minLineLength,maxLineGap)
        result_P = img.copy()
        for i_P in linesP:
            for x1, y1, x2, y2 in i_P:
                cv.line(result_P, (x1, y1), (x2, y2), (0, 255, 0), 3)
        img = result_P
    elif num == 34:  # canny边缘检测
        blur = cv.GaussianBlur(img, (3, 3), 0)
        image = cv.cvtColor(blur, cv.COLOR_BGR2GRAY)
        gradx = cv.Sobel(image, cv.CV_16SC1, 1, 0)
        grady = cv.Sobel(image, cv.CV_16SC1, 0, 1)
        img = cv.Canny(gradx, grady, 50, 150)
    elif num == 35:  # 图像增强
        CRH = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        CRH = CRH.astype('float')
        row, column = CRH.shape
        gradient = np.zeros((row,column))
        for x in range(row - 1):
            for y in range(column - 1):
                gx = abs(CRH[x + 1, y] - CRH[x, y])
                gy = abs(CRH[x, y + 1] - CRH[x, y])
                gradient[x, y] = gx + gy
        sharp = CRH+gradient
        sharp = np.where(sharp > 255, 255, sharp)
        sharp = np.where(sharp < 0, 0, sharp)
        gradient = gradient.astype('uint8')
        img = sharp.astype('uint8')

    elif num == 36:  # Roberts算子提取图像边缘
        grayImage = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        kernelx = np.array([[-1, 0], [0, 1]], dtype=int)
        kernely = np.array([[0, -1], [1, 0]], dtype=int)
        x = cv.filter2D(grayImage, cv.CV_16S, kernelx)
        y = cv.filter2D(grayImage, cv.CV_16S, kernely)
        absX = cv.convertScaleAbs(x)
        absY = cv.convertScaleAbs(y)
        img = cv.addWeighted(absX, 0.5, absY, 0.5, 0)
    elif num == 37:  # Prewitt 算子提取图像边缘
        grayImage = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        x=cv.Sobel(grayImage, cv.CV_16S, 1, 0)
        y=cv.Sobel(grayImage, cv.CV_16S, 0, 1)
        absX=cv.convertScaleAbs(x)
        absY=cv.convertScaleAbs(y)
        img=cv.addWeighted(absX,0.5,absY,0.5,0)
    elif num == 38:  # Laplacian算子提取图像边缘
        grayImage = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        grayImage = cv.GaussianBlur(grayImage, (5, 5), 0, 0)
        dst = cv.Laplacian(grayImage, cv.CV_16S,ksize=3)
        img = cv.convertScaleAbs(dst)
    elif num == 39:  # LoG边缘提取
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        image = cv.copyMakeBorder(img, 2, 2, 2, 2, borderType=cv.BORDER_REPLICATE)
        image = cv.GaussianBlur(image, (3, 3), 0, 0)
        m1 = np.array([[0, 0, -1, 0, 0], [0, -1, -2, -1, 0], [-1, -2, 16, -2, -1], [0, -1, -2, -1, 0], [0, 0, -1, 0, 0]])
        rows=image.shape[0]
        cols=image.shape[1]
        image1=np.zeros(image.shape)
        for k in range(0, 2):
            for i in range(2, rows-2):
                for j in range(2, cols-2):
                    image1[i, j] = np.sum((m1*image[i - 2:i + 3, j - 2:j + 3, k]))
        img = cv.convertScaleAbs(image1)
    elif num==40: # 风格转换-糖果
        img = ut.trans2candy(img)
    elif num==41: # 风格转换-星空
        img = ut.trans2star(img)
    elif num==42: # 风格转换-毕加索
        img = ut.trans2bjs(img)
    elif num==43: # 风格转换-缪斯
        img = ut.trans2ms(img)
    elif num==44: # 风格转换-马赛克
        img = ut.trans2mosaic(img)
    elif num==45: # 风格转换-神奈川冲浪里
        img = ut.trans2snc(img)
    elif num==46: # 风格转换-达达主义
        img = ut.trans2dd(img)
    elif num==47: # 风格转换-呐喊
        img = ut.trans2nh(img)
    elif num==48: # 风格转换-羽毛
        img = ut.trans2fea(img)
    elif num == 49: # 图像修复
        img = ut.colorizer(img)

    cv.imencode(".png", img)[1].tofile(r'./tmp/draw/{}.{}'.format(file_name, ext))

    return data_path, file_name


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




def trans_1(image,data_path,ext,file_name):

    # 加载模型
    net = cv.dnn.readNetFromTorch('.\models\candy.t7')#选择一个模型的地址
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)#创建后端
    # 读取图片
    #image = cv.imread(data_path)
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
def trans_2(image,data_path,ext,file_name):

    # 加载模型
    net = cv.dnn.readNetFromTorch('.\models\starry_night.t7')#选择一个模型的地址
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)#创建后端
    # 读取图片
    #image = cv.imread(data_path)
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
def trans_3(image,data_path,ext,file_name):

    # 加载模型
    net = cv.dnn.readNetFromTorch('.\models\composition_vii.t7')#选择一个模型的地址
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)#创建后端
    # 读取图片
    #image = cv.imread(data_path)
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
def trans_4(image,data_path,ext,file_name):

    # 加载模型
    net = cv.dnn.readNetFromTorch('.\models\la_muse.t7')#选择一个模型的地址
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)#创建后端
    # 读取图片
    #image = cv.imread(data_path)
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
def trans_5(image,data_path,ext,file_name):

    # 加载模型
    net = cv.dnn.readNetFromTorch('.\models\mosaic.t7')#选择一个模型的地址
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)#创建后端
    # 读取图片
    #image = cv.imread(data_path)
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
def trans_6(image,data_path,ext,file_name):

    # 加载模型
    net = cv.dnn.readNetFromTorch('.\models\\the_wave.t7')#选择一个模型的地址
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)#创建后端
    # 读取图片
    #image = cv.imread(data_path)
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
def trans_7(image,data_path,ext,file_name):

    # 加载模型
    net = cv.dnn.readNetFromTorch(r'.\models\udnie.t7')#选择一个模型的地址
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)#创建后端
    # 读取图片
    #image = cv.imread(data_path)
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

def trans_8(image,data_path,ext,file_name):

    # 加载模型
    net = cv.dnn.readNetFromTorch('.\models\\the_scream.t7')#选择一个模型的地址
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)#创建后端
    # 读取图片
    #image = cv.imread(data_path)
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

def trans_9(image,data_path,ext,file_name):

    # 加载模型
    net = cv.dnn.readNetFromTorch('.\models\\feathers.t7')#选择一个模型的地址
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)#创建后端
    # 读取图片
    #image = cv.imread(data_path)
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


def colorizer(image, data_path,ext,file_name):
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
    # image = cv.imread(image)  # 读取图像文件
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
    return colorized