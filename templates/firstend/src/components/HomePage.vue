<template>
    <div class="content">
        <div class="show">
            <div class="carousel-container">
                <el-carousel :interval="0" arrow="always" class="el-carousel">
                    <el-carousel-item v-for="item in pictures" :key="item.id" class="el-carousel-item">
                        <img :src="item.path" :alt="`Image ${item.id}`" class="carousel-image">
                    </el-carousel-item>
                </el-carousel>
            </div>
        </div>
        <div class="info">
            <div class="top">
                <span class="title1">PixEditPro</span>
                <span class="title2">功能简介</span>
            </div>
            <div class="task">
                <div class="t1">
                    <img :src="point" class="tag">
                    <span class="text">实现了数字图像处理的40个基础功能</span>
                </div>
                <div class="t1">
                    <img :src="point" class="tag">
                    <span class="text">实现了针对黑白图像的图像上色，实现黑白老照片修复功能</span>
                </div>
                <div class="t1">
                    <img :src="point" class="tag">
                    <span class="text">实现了针对9种艺术画风格的图像风格迁移效果</span>
                </div>
            </div>
            <div class="advan">
                <span class="a1">我们的</span>
                <span class="a2">优势</span>
            </div>
            <div class="t1">
                <span class="text">   实现功能多</span>
                <span class="text">   处理速度快</span>
                <span class="text">   输出结果好</span>
            </div>
        </div>
    </div>
</template>



<script>
import {StyleCategories} from "@/utils/function";
export default {
    data() {
        return {
            point: require("@/static/point.png"),
            pictures: [
                { id: 0, path: require('@/static/img/scenary1.jpg') },
                { id: 1, path: require('@/static/img/style2.png') },
                { id: 2, path: require('@/static/img/scenary2.jpg') },
            ]
        };
    },
    methods: {
        selectItem(item) {
            console.log('Selected Item:', item);
        }
    }
};
</script>


<style scoped>
.content{
    position: fixed;
    top: 150px;
    left: 180px;
    display: flex;
    flex-direction: row;
}
.info{
    background-color: rgba(255, 255, 255, 0.5);
    width: 500px;
    height: 300px;
    margin-left: 50px;
    border-radius: 10px;
    padding: 20px;
}
.title1{
    color: #FFFFFF;
    font-size: 20px;
    font-weight: bold;
}
.title2{
    margin-left: 10px;
    color: #ffd04b;
    font-size: 20px;
    font-weight: bold;
    letter-spacing: 0.3px;
}
.task{
    margin-top: 10px;
}
.tag{
    width: 25px;
    height: 25px;
}
.text{
    color: #72767b;
    font-size: 17px;
}
.t1{
    margin-top: 15px;
}
.advan{
    margin-top: 15px;
}
.a1{
    font-size: 20px;
    font-weight: bold;
    letter-spacing: 1px;
    color: #FFFFFF;
}
.a2{
    color: #ef008f;
    font-size: 20px;
    font-weight: bold;
    letter-spacing: 2px;
}
.show{
}
.title{
    color: #FFFFFF;
}

.carousel-container {
    margin-top: 30px;
    width: 550px;
    height: 300px;
}

.el-carousel {
    height: 100%;
}

.el-carousel-item {
    display: flex;
    justify-content: center;
    align-items: center;
    height: 100%;
}

.carousel-image {
    width: auto;  /* 保持图片宽度自动 */
    height: 100%; /* 让图片高度撑满容器 */
    object-fit: contain; /* 确保图片按比例缩放 */
}
</style>
