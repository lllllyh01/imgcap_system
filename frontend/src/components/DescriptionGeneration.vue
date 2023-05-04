<template>
    <h1>欢迎，lyh！</h1>
    <el-upload
    class="upload_img"
    :show-file-list="false"
    :on-success="handleImgSuccess"
    action="https://jsonplaceholder.typicode.com/posts/"
    :before-upload="beforeUpload"
    multiple
  >
    <el-icon class="el-icon__upload"><upload-filled /></el-icon>
    <div class="el-upload__text">
      拖动图片至此 或 <em>点击上传图片</em>
    </div>
    <template #tip>
      <div class="el-upload__tip">
        请上传JPG或PNG格式图片
      </div>
    </template>
  </el-upload>
  <img v-if="imageUrl" :src="imageUrl" />
  <el-button class="gen_button" @click="generate()" disabled="imageUrl">生成描述</el-button>
  <div>
    <el-input class="des_box" type="textarea" :rows="8" placeholder="描述内容" v-model="description"/>
  </div>
</template>

<script>
  import { ref } from 'vue'
  import { ElMessage } from 'element-plus'
  // eslint-disable-next-line no-unused-vars
  import { UploadFilled } from '@element-plus/icons-vue'
  export default {
    name: 'DescriptionGeneration',
    data () {
      return {
        imageUrl: ref(''),
      }
    },
    methods: {
      handleImgSuccess(res, file) {
        console.log(res);
        console.log(file);
        if(res.success == true) {
          ElMessage.success("图片上传成功");
          this.imageUrl = URL.createObjectURL(file.raw);
        } else {
          ElMessage.error("上传失败，请重试")
        }
      },
      beforeUpload(file) {
        const isJPGPNG = file.type === 'image/jpeg' || file.type === 'image/png'

        if (!isJPGPNG) {
          ElMessage.error('请上传JPG或PNG格式图片')
        }
        return isJPGPNG
      },
      generate() {
        console.log("in generate function")
      }
    },
  }
</script>
<style>
  .upload_img .el-upload {
    border: 1px dashed #d9d9d9;
    width: 500px;
    height: 200px;
    border-radius: 6px;
    cursor: pointer;
    position: relative;
    overflow: hidden;
  }
  .upload_img .el-upload:hover {
    border-color: #409eff;
  }
  .el-icon__upload {
    font-size: 100px;
    margin-top: 40px;
    vertical-align: middle;
    text-align: center;
  }
  .el-upload__text {
    font-size: 13px;
  }
  .gen_button {
    margin-top: 10px;
  }
  .des_box {
    width: 500px;
    margin-top: 20px;
  }
</style>