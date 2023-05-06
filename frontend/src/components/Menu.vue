<template>
  <div class="menu">
    <el-row style="background-color: #304156; height: 60px;">
      <div style="height: 60px; line-height: 60px; margin-left: 30px; margin-right: 100px; color: #f4f4f5;">
        <span style="font-size: 30px; font-weight: bolder; margin-right: 20px;">图像描述生成系统</span>
        <span style="font-size: 20px;">欢迎，lyh！</span>
      </div>
      <el-menu :default-active="activeIndex"
      mode="horizontal"
      text-color="#f4f4f5"
      active-text-color="#409EFF"
      background-color="#304156"
      style="margin-right: 240px; height: 60px;"
      router
      unique-opened>
        <el-menu-item index="/description_generation">
          <i class="el-icon-s-management"></i>报告生成
        </el-menu-item>
        <el-menu-item index="/person">
          <i class="el-icon-reading"></i>个人信息
        </el-menu-item>
        <el-menu-item index="/history">
          <i class="el-icon-document"></i>历史记录
        </el-menu-item>
      </el-menu>
      <div style="height: 60px; line-height: 60px; margin: auto; margin-left: 240px; margin-right: 10px;">
        <el-button style="margin: auto;">退出登录</el-button>
      </div>
    </el-row>
    <el-scrollbar>
      <keep-alive>
        <router-view></router-view>
      </keep-alive>
    </el-scrollbar>
  </div>
</template>

<script>
  export default {
    name: 'Menu',
    data () {
      return {
      }
    },
    methods: {
      back() {
        console.log("in back function")
      }
    }
  }
</script>

<style>
  .menu {
    height: 100%;
  }
  .el-menu--horizontal .el-menu .el-menu-item {
    font-size: 25px;
  }
  .col {
    background-color: #324057;
    height: 100% !important;
    min-height: 100% !important;
    margin-top: 0px;
    margin-left: 0px;
    position: fixed;
    width: 16%;
  }
  .content {
    height: 100%;
    overflow: auto;
    margin-right: 0px
  }
  /* .learnMore {
    font-size: 17px;
    font-weight: bold;
    justify-items: end;
    color: #4083ffb4;
  }
  .title {
    justify-content: start;
    color:rgb(75, 75, 75);
  } */
  .welcome_title {
    font-size: medium;
    margin: 10px;
  }
  .logout_botton {
    text-align: right;
  }
</style>