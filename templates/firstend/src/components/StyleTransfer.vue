<template>
    <div class="content">
        <div class="title">
            <img :src="icon" class="icon">
            <span class="info">风格变换</span>
        </div>
        <div class="toolbar">
            <el-radio-group v-model="currentId" @change="handleChange(currentId)">
                <el-radio v-for="item in items" :label="item.id" :key="item.id">
                    {{ item.name }}
                </el-radio>
            </el-radio-group>
        </div>
        <div v-if="currentImage" class="image-box">
            <img :src="currentImage" alt="Selected Image" class="responsive-image" />
            <span class="text">Style</span>
        </div>
        <div class="workspace">
            <el-dialog title="图片正在处理中,请稍后..." :visible.sync="dialogTableVisible" :show-close="false"
                       :close-on-press-escape="false" :append-to-body="true" :close-on-click-modal="false" :center="true"
                       custom-class="custom-dialog">
                <el-progress :percentage="percentage" color="#dba23c"></el-progress>
            </el-dialog>
            <div class="center-wrapper">
                <el-button type="warning" round icon="el-icon-upload"
                           class="upload-button" v-on:click="true_upload">
                    上传图像<input ref="upload" style="display: none" name="file" type="file"
                                   @change="update" />
                </el-button>
                <el-card class="box-card">
                    <div class="image-container">
                        <div class="demo-image__preview" v-loading="loading" element-loading-text="上传图片中"
                             element-loading-spinner="el-icon-loading">
                            <el-image :src="url_1" class="image_1" :preview-src-list="srcList">
                                <div slot="error">
                                    <div slot="placeholder" class="upload-button-container">
                                        <span>请上传图片</span>
                                    </div>
                                </div>
                            </el-image>
                            <div class="img_info_1">
                                <span style="color: white; letter-spacing: 9px; font-size: 20px;">原始图像</span>
                            </div>
                        </div>

                        <div class="demo-image__preview" v-loading="loading" element-loading-text="处理中"
                             element-loading-spinner="el-icon-loading">
                            <el-image :src="url_2" class="image_1" :preview-src-list="srcList1">
                                <div slot="error">
                                    <div slot="placeholder" class="error">{{ wait_return }}</div>
                                </div>
                            </el-image>
                            <div class="img_info_1">
                                <span style="color: white; letter-spacing: 9px; font-size: 20px;">处理结果</span>
                            </div>
                        </div>
                    </div>
                </el-card>
            </div>
        </div>
    </div>

</template>


<script>
import axios from "axios";
import {StyleCategories} from "@/utils/function";


export default {
    data() {
        return {
            icon: require('@/static/transfer-icon.png'),
            items: StyleCategories,
            currentId: null,
            currentImage: null,
            server_url: "http://127.0.0.1:5000",
            url_1: "", url_2: "",
            srcList: [], srcList1: [],
            url: "",
            visible: false,
            wait_return: "等待上传", wait_upload: "等待上传",
            loading: false,
            showbutton: true, percentage: 0,
            fullscreenLoading: false,
            dialogTableVisible: false,
        };
    },
    created: function () {
        this.$watch('$route.params.id',(newId, oldId) =>{
            this.resetData()
        })
        console.log(" :key=\"$route.params.id\"", this.$route.params.id)
    },
    methods: {
        true_upload() {
            this.$refs.upload.click();
        },
        true_upload2() {
            this.$refs.upload2.click();
        },
        handleChange(id) {
            this.currentId = id;
            console.log(this.currentId)
            const selectedItem = this.items.find(item => item.id === this.currentId);
            if (selectedItem) {
                this.currentImage = selectedItem.path;
                console.log(this.currentImage)
            } else {
                this.currentImage = null;
            }
        },
        // 获得目标文件
        getObjectURL(file) {
            var url = null;
            if (window.createObjcectURL !== undefined) {
                url = window.createOjcectURL(file);
            } else if (window.URL !== undefined) {
                url = window.URL.createObjectURL(file);
            } else if (window.webkitURL !== undefined) {
                url = window.webkitURL.createObjectURL(file);
            }
            return url;
        },
        selectMethod(id) {
            this.currentId = id;
            console.log(this.currentId)
        },
        // 上传文件
        update(e) {
            this.percentage = 0;
            this.dialogTableVisible = true;
            this.url_1 = "";
            this.url_2 = "";
            this.srcList = [];
            this.srcList1 = [];
            this.wait_return = "";
            this.wait_upload = "";
            this.fullscreenLoading = true;
            this.loading = true;
            this.showbutton = false;
            let file = e.target.files[0];
            this.url_1 = this.$options.methods.getObjectURL(file);
            let param = new FormData(); //创建form对象
            param.append("file", file, file.name); //通过append向form对象添加数据
            var timer = setInterval(() => {
                this.myFunc();
            }, 30);
            let config = {
                headers: { "Content-Type": "multipart/form-data" },
            }; //添加请求头 DONE 这里获取this.id
            // console.log("this.id",this.id);
            axios.post(this.server_url + `/upload/${this.currentId}`, param, config)
                .then((response) => {
                    this.percentage = 100;
                    clearInterval(timer);
                    this.url_1 = response.data.image_url;
                    this.srcList.push(this.url_1);
                    this.url_2 = response.data.draw_url;//处理后的
                    this.srcList1.push(this.url_2);
                    this.fullscreenLoading = false;
                    this.loading = false;
                    this.dialogTableVisible = false;
                    this.percentage = 0;
                    this.notice1();
                });
        },
        myFunc() {
            if (this.percentage + 33 < 99) {
                this.percentage = this.percentage + 33;
            } else {
                this.percentage = 99;
            }
        },
        drawChart() {},
        notice1() {
            this.$notify({
                title: "处理成功",
                message: "点击图片可以查看大图",
                duration: 0,
                type: "success",
            });
        },
        resetData(){
            window.location.reload()
        },
    },
    mounted() {
        this.drawChart();
    },
};
</script>

<style scoped>
.title{
    display: flex;
    flex-direction: row;
}
.info{
    color: #FFFFFF;
    font-size: 20px;
    font-weight: bold;
    letter-spacing: 1px;
    margin-left: 10px;
}
.icon {
    width: 27px;
    height: 23px;
}
.content{
    position: fixed;
    top: 100px;
    left: 100px;
}
.toolbar{
    margin-top: 10px;
    background-color: rgba(255, 255, 255, 0.5);
    padding: 10px;
    width: 1200px;
    border-radius: 10px;
}
.image-box {
    margin-top: 25px;
    margin-left: 10px;
    position: fixed;
    z-index: 99;
    width: 200px; /* 你可以根据需要调整宽度 */
    height: auto; /* 保持高度自动调整 */
    overflow: hidden; /* 隐藏溢出部分 */
    border-radius: 20px;
    box-shadow: 0 4px 8px rgba(0, 0, 0.1, 0.5);
}
.text{
    position: absolute;
    z-index: 0;
    bottom: 20px;
    left: 60px;
    color: #ffffff;
    font-size: 30px;
    font-weight: bold;
}
.responsive-image {
    width: 100%;
    height: auto; /* 保持宽高比 */
    object-fit: cover; /* 填充容器并保持图像比例 */
}
.center-wrapper {
    display: flex;
    justify-content: center;
    align-items: center;
}

.box-card {
    margin-top: 10px;
    width: 1210px;
    height: 500px;
    border-radius: 8px;
    background-color: rgba(255, 255, 255, 0.5);
}
.workspace{
    position: fixed;
    top: 180px;
}

.image-container {
    display: flex;
    justify-content: space-between;
    width: 100%;
    height: 100%;
}

.demo-image__preview {
    display: flex;
    flex-direction: column;
    align-items: center;
    width: 48%;
    height: 100%;
    position: relative;
}

.image_1 {
    margin-top: 24px;
    width: 450px;
    height: 350px;
    background: #ffffff;
    box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.1);
    display: flex;
    justify-content: center;
    align-items: center;
    border-radius: 10px;
}

.upload-button-container {
    display: flex;
    justify-content: center;
    align-items: center;
    width: 100%;
    height: 100%;
}

.upload-button {
    position: absolute;
    background-image: linear-gradient(30deg,  #e2d054, #f1ed13);
    border-color: #e2d054;

}

.img_info_1 {
    height: 70px;
    width: 450px;
    text-align: center;
    background-image: linear-gradient(30deg,  #e2d054, #f1ed13);
    line-height: 70px;
    display: flex;
    justify-content: center;
    align-items: center;
    border-radius: 10px;
}

.error {
    margin: 100px auto;
    width: 50%;
    padding: 10px;
    text-align: center;
}

.download-button {
    display: block;
    margin: 20px auto;
}

.custom-dialog {
    border-radius: 20px;
}

/* 自定义加载状态文本样式 */
.el-loading-mask .el-loading-text {
    color: #dba23c;
    /* 修改颜色 */
    font-size: 18px;
    /* 修改字体大小 */
    font-weight: bold;
    /* 修改字体粗细 */
}

/* 自定义加载动画颜色 */
.el-loading-mask .el-loading-spinner {
    border-top-color: #dba23c;
    /* 修改旋转图标颜色 */
}
</style>
