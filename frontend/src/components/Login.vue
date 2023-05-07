<template>
  <h1 style="font-size: 25px; text-align: center; padding-top: 20px; padding-bottom: 0px;">图像描述生成系统</h1>
  <div class="login_box">
    <div class="login_form_div">
      <el-form :class="login_form" :model="loginForm" :rules="loginRules" label-position="top" label-width="380px">
        <el-form-item prop="userName" label="用户名">
          <el-input v-model="loginForm.userName" />
        </el-form-item>
        <el-form-item prop="password" label="密码">
          <el-input v-model="loginForm.password" />
        </el-form-item>
        <el-form-item>
          <el-button type="primary" @click="login">登录</el-button>
        </el-form-item>
      </el-form>
    </div>
  </div> 
  <div class="register_box">
    <span style="font-size: 17px;">没有账号？
      <span @click="goRegister" style="color: #659DFF;
      font-style: italic; font-size: 20px;">注册</span>
    </span>
  </div>
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
        this.$router.push("/menu")
      } else {
        this.$message.error("用户名或密码错误")
      }
    },
    goRegister () {
      this.$router.push("/register")
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
  .login_box {
    width: 400px;
    height: 300px;
    margin: auto;
    border: solid;
    border-width: 2px;
    box-shadow: 0px 0px 5px rgba(0,0,0,0.1);
    border-radius: 10px;
    border-color: gainsboro;
    position: relative;
  }
  .login_form_div {
    width: 380px;
    position: absolute;
    padding: 10px 10px;
  }
  .register_box {
    width: 400px;
    height: 50px;
    border: solid;
    border-width: 2px;
    box-shadow: 0px 0px 5px rgba(0,0,0,0.1);
    border-radius: 10px;
    border-color: gainsboro;
    margin: auto;
    margin-top: 13px;
    line-height: 50px;
  }
</style>
