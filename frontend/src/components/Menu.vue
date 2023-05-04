<template>
    <div class="Menu" style="height: 100%; margin: 0%;">
        <el-row style="margin: 0%; height: 100%;">
        <el-col :span="4" class="col" style="height: 100%; min-height: 100%; width: 16%;">
            <el-menu :default-active="defaultActive"
            style="height: 100%;border-right:0px; display: block; min-height: 100%;"
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
            <el-scrollbar>
            <keep-alive>
                <router-view></router-view>
            </keep-alive>
        </el-scrollbar>
        </el-col>
        </el-row>
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
    }
  }
</script>

<style scoped>
    .col {
      background-color: #324057;
      height: 100vh !important;
      min-height: 100vh !important;
      margin-top: 0px;
      margin-left: 0px;
      position: fixed;
    }
    .content {
      height: 100%;
      overflow: auto;
      margin-right: 0px
    }
    .learnMore {
      font-size: 17px;
      font-weight: bold;
      justify-items: end;
      color: #4083ffb4;
    }
    .title {
      justify-content: start;
      color:rgb(75, 75, 75);
    }
</style>