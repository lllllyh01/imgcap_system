<template>
  <div class="menu">
      <el-col :span="4" class="col">
        <el-menu :default-active="defaultActive" class="el-menu-vertical-demo"
        text-color="#bfcbd9"
        background-color="#324057"
        router
        unique-opened
        >
          <el-menu-item index="/description_generation">
            <i class="el-icon-s-management"></i>报告生成
          </el-menu-item>
          <el-menu-item index="/person">
            <i class="el-icon-reading"></i>个人信息
          </el-menu-item>
            <!-- <el-submenu index="/project">
                <template slot="title"><i class="el-icon-reading"></i>项目</template>
                <el-menu-item index="/back_ground">研究背景</el-menu-item>
                <el-menu-item index="/research_content">研究内容</el-menu-item>
                <el-menu-item index="/technology">技术路线</el-menu-item>
                <el-menu-item index="/goal">目标与成果</el-menu-item>
            </el-submenu> -->
          <el-menu-item index="/history">
            <i class="el-icon-document"></i>历史记录
          </el-menu-item>
        </el-menu>
      </el-col>

      <el-col :span="20" class="content" style="margin-left: 16.5%; height: 100%;overflow: auto;width:84%; margin-right: 0%">
        <el-row>
          <el-page-header @back="back">
            <template #content>
              <span class="welcome_title">欢迎，lyh！</span>
            </template>
            <template #extra>
              <el-button>退出登录</el-button>
            </template>
          </el-page-header>
        </el-row>
        <el-scrollbar>
          <keep-alive>
            <router-view></router-view>
          </keep-alive>
        </el-scrollbar>
      </el-col>
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