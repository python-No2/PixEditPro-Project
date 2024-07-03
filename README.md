# PixEditPro

2023-2024学年秋季学期数字图像处理期末项目

### 项目功能列表

1.  40个数字图像基本操作，包括：椒盐噪声、均值平滑、图像开运算、腐蚀、膨胀等操作
2.  黑白图像上色，实现老照片修复功能
3.  实现9种艺术画风格的图像风格迁移功能，包括：糖果，星空 ，毕加索，缪斯，马赛克，神奈川冲浪里，抽象主义，呐喊，羽毛这9种艺术风格

### 技术栈

#### 前端技术栈

Vue + Axios + ElementUI + Vue Router

- 前端包管理工具：npm
- 前端构建工具：Webpack

#### 后端技术栈

Flask + flask-cors

### 项目目录结构

#### 前端

```
|——src

|		|——assets

|				 |——style.css

|		|——components

|		|		|——BasicTask.vue    基础功能页面

|		|		|——Colorizer.vue      图像修复页面

|		|		|——Header.vue    导航栏

|		|		|——HomePage.vue    首页

|		|		|——StyleTransfer.vue    风格迁移页面

|		|——router

|		|		|——index.js    页面路由配置文件

|		|——static

|		|		|——img   存放一些展示图片

|		|		|——style-type    存放一些风格模板图片

|		|——theme    elment-ui组件样式配置文件

|		|——utils

|		|		|——function.js  存放一些函数和数组

|		|——App.vue  

|		|——Component.vue

|		|——main.js  配置文件
```



#### 后端

```
|——PixEditPro

|		|——core    核心功能文件

|		|		|——\__init__.py

|		|		|——main.py

|		|		|——process.py     主要功能处理文件

|		|		|——utils.py	   相关函数

|		|——models		一些训练模型

|		|——templates		网页文件

|		|——tmp		用于存放处理前和处理后的图片

|		|		|——ct

|		|		|——draw

|		|——uploads	 用于存放上传给前端的图片

|		|		|——readmePic		存放README文件里的图片

|		|——app.py		项目入口点，定义路由

|		|——config.py 		项目配置文件

|		|——README.md
```



### 相关实现技术

- 快速神经网络

- 卷积层

- 深度学习

  

# Data preparation

由于125MB太大，无法上传，本项目运行之前需要从此位置下载预先训练的数据。并将此文件放入`models`文件夹中，作为图像修复的模型参数。

https://www.dropbox.com/s/dx0qvhhp5hbcx7z/colorization_release_v2.caffemodel?dl=1



# Get started

### 前端部分

#### Install Dependencies

```bash
npm install
```

#### Develop

```bash
npm run dev
```

项目将运行在 http://127.0.0.1:8080 网址



### 后端部分

#### Run

```
python app.py
```

在终端运行上述指令



# Feature demonstration

### 一、基础功能

​	页面实现如下：

<img src="uploads\readmePic\basic-page.png" alt="image-20240703053323116" style="zoom:40%;" />

具体的基础功能展示请在报告中查看

### 二、老照片修复

​	页面实现如下：

<img src="uploads\readmePic\colorizer-page.png" alt="image-20240703053456051" style="zoom:80%;" />

​	修复效果展示如下：

<img src="uploads\readmePic\color1.png" alt="image-20240703054808530" style="zoom:60%;" />

<img src="uploads\readmePic\color2.png" alt="image-20240703055838359" style="zoom:50%;" />

修复得到的结果非常自然！！

### 三、风格迁移

​	页面实现如下：

<img src="uploads\readmePic\transfer-page.png" alt="image-20240703060101950" style="zoom:85%;" />

​	具体可实现的风格如下：

