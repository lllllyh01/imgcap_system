<template>
    <h1 style="font-size: 25px; text-align: center; padding-top: 20px; padding-bottom: 0px;">图像描述生成系统</h1>
    <div class="register_box">
      <h1 style="font-size: 15px; text-align: center; padding-top: 15px;">注册</h1>
      <div class="register_form_div">
        <el-form :class="register_form" :model="registerForm" :rules="registerRules" label-width="380px">
          <el-form-item prop="userName" label="用户名">
            <el-input v-model="registerForm.userName" />
          </el-form-item>
          <el-form-item prop="passwd" label="密码">
            <el-input v-model="registerForm.passwd" />
          </el-form-item>
          <el-form-item prop="confirm_passwd" label="确认密码">
            <el-input v-model="registerForm.confirm_passwd" />
          </el-form-item>
          <el-form-item label="真实姓名">
            <el-input v-model="registerForm.real_name" />
          </el-form-item>
          <el-form-item label="性别">
            <el-option label="男" value="man" />
            <el-option label="女" value="woman" />
            <el-option label="" value="none" />
          </el-form-item>
          <el-form-item prop="age" label="年龄">
            <el-input v-model="registerForm.age" />
          </el-form-item>
          <el-form-item>
            <el-button type="primary" @click="register">注册</el-button>
          </el-form-item>
        </el-form>
      </div>
    </div> 
    <div class="go_login_box">
      <span style="font-size: 17px;">已有账号？
        <span @click="goLogin" style="color: #659DFF;
        font-style: italic; font-size: 20px;">返回登录</span>
      </span>
    </div>
  </template>
  
  <script>
  export default {
    name: 'Register',
    data () {
      var confirmPasswd = (rule, value, callback) => {
        if (value !== this.registerForm.passwd) {
            callback(new Error("两次密码输入不一致！"))
        } else {
            callback()
        }
      };
      return {
        registerForm: {
          userName: '',
          passwd: '',
          confirm_passwd: '',
          real_name: '',
          gender: '',
          age: 0
        },
        registerRules: {
          userName: [
            {
              required: true,
              message: '用户名不能为空',
              trigger: 'blur'
            }
          ],
          passwd: [
            {
              required: true,
              message: '密码不能为空',
              trigger: 'blur'
            }
          ],
          confirm_passwd: [
            {
                validator: confirmPasswd,
                trigger: 'blur'
            }
          ],
          age: [
            {
                required: false,
                type: 'number',
                message: "年龄必须是数值"
            }
          ]
        }
      }
    },
    methods: {
      register () {
        this.$router.push("/login")
      },
      goLogin () {
        this.$router.push("/login")
      }
    }
  }
  </script>
  
  <style scoped>
    .register_box {
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
    .register_form_div {
      width: 380px;
      position: absolute;
      padding: 10px 10px;
    }
    .go_login_box {
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