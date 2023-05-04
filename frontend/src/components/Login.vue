<template>
  <h1>图像描述生成系统</h1>
  <div class="login">
    
    <el-form :class="login_form" :model="loginForm" :rules="loginRules">
      <el-form-item prop="userName">
        <el-input placeholder="用户名" v-model="loginForm.userName" />
      </el-form-item>
      <el-form-item prop="password">
        <el-input placeholder="密码" v-model="loginForm.password" />
      </el-form-item>
    </el-form>
  </div> 
  <el-button type="primary" @click="login">登录</el-button>
  <!-- <nav>
    <router-link to="/register">
      <span style="font-size: 14px">新用户注册</span>
    </router-link>
  </nav> -->
</template>

<script>
// import Cookies from "js-cookie";
// // import api from "../api/api_export";
// import axios from 'axios'
// import ElAlert from "element-plus"
export default {
  name: 'Login',
  data () {
    return {
      loginForm: {
        userName: '',
        password: '',
      },
      loginRules: {
        userName: [
          {
            required: true,
            message: '用户名不能为空',
            trigger: 'blur'
          }
        ],
        password: [
          {
            required: true,
            message: '密码不能为空',
            trigger: 'blur'
          }
        ]
      }
    }
  },
  methods: {
    login () {
      if(this.loginForm.userName === "lyh" && this.loginForm.password === "123") {
        this.$router.push("/")
      } else {
        this.$message.error("用户名或密码错误");
      }
    }
    // clkicLogin () {
    //   const that = this
    //   axios
    //     .post('http://127.0.0.1:5000', this.login_form)
    //     .then(function (response) {
    //       if (response.data.code == 0) {
    //         sessionStorage.setItem('accessToken', response.data.session)
    //         that.$router.push({ path: '/home' })
    //       } else if (response.data.code == 1002) {
    //         console.log('用户名不存在')
    //       } else if (response.data.code == 1003) {
    //         console.log('密码错误')
    //       }
    //     })
    //     .catch(function (error) {
    //       console.log(error)
    //     })
    // }
  }
}
</script>

<style scoped>
  .login_form {
    width: 60%;
  }
</style>
