<template>
  <h1 style="font-size: 25px; text-align: center; padding-top: 20px; padding-bottom: 0px;">图像描述生成系统</h1>
  <div class="register_box">
    <h1 style="font-size: 20px; text-align: center;">注册</h1>
    <div class="register_form_div">
      <el-form :class="register_form" :model="registerForm" :rules="registerRules" label-width="80px">
        <el-form-item prop="userName" label="用户名">
          <el-input class="input_box" v-model="registerForm.userName" />
        </el-form-item>
        <el-form-item prop="passwd" label="密码">
          <el-input class="input_box" type="password" v-model="registerForm.passwd" />
        </el-form-item>
        <el-form-item prop="confirm_passwd" type="password" label="确认密码">
          <el-input class="input_box" type="password" v-model="registerForm.confirm_passwd" />
        </el-form-item>
        <el-form-item label="真实姓名">
          <el-input class="input_box" v-model="registerForm.real_name" />
        </el-form-item>
        <el-form-item label="性别">
          <el-select class="input_box" v-model="registerForm.gender" placeholder="请选择您的性别">
            <el-option label=" " value="none" />
            <el-option label="男" value="man" />
            <el-option label="女" value="woman" />
          </el-select>
        </el-form-item>
        <el-form-item prop="age" label="年龄">
          <el-input class="input_box" v-model="registerForm.age" />
        </el-form-item>
        <el-form-item style="margin-left: -50px;">
          <el-button type="primary" @click="register" style="text-align: center;">注册</el-button>
        </el-form-item>
      </el-form>
    </div>
  </div> 
  <div class="go_register_box">
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
            required: true,
            validator: confirmPasswd,
            trigger: 'blur'
          }
        ],
        age: [
          {
              required: false,
              type: 'number',
              message: "年龄必须是数值",
              transform(value) { return Number(value) }
          }
        ]
      }
    }
  },
  methods: {
    register () {
      if (this.registerForm.userName === "lyh" && this.registerForm.passwd === "123") {
        this.$message.success("注册成功")
        this.$router.push("/login")
      } else if (this.registerForm.userName === "lyh" && this.registerForm.passwd === "456") {
        this.$message.error("用户已存在")
      }
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
    height: 500px;
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
    text-align: center;
  }
  .input_box {
    width: 300px;
    display: inline-block;
  }
  .go_register_box {
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