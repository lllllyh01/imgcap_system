<template>
    <Menu></Menu>
    <div>
    <div class="info_row">
        <span class="info_key">用户名：</span>
        <span class="info_box">{{username}}</span>
        <el-button class="modify_button" type="primary" @click="modify_form_visible = true">修改</el-button>
    </div>
    <div class="info_row">
        <span class="info_key">密码：</span>
        <span class="info_box">*****</span>
        <el-button class="modify_button" type="primary" @click="this.modify_form_visible = true">修改</el-button>
    </div>
    <div class="info_row">
        <span class="info_key">真实姓名：</span>
        <span class="info_box">{{real_name}}</span>
        <el-button class="modify_button" type="primary" @click="modify_form_visible = true">修改</el-button>
    </div>
    <div class="info_row">
        <span class="info_key">性别：</span>
        <span class="info_box">{{gender}}</span>
        <el-button class="modify_button" type="primary" @click="modify_form_visible = true">修改</el-button>
    </div>
    <div class="info_row">
        <span class="info_key">年龄：</span>
        <span class="info_box">{{age}}</span>
        <el-button class="modify_button" type="primary" @click="modify_form_visible = true">修改</el-button>
    </div>
</div>
    <el-dialog title="修改密码" v-model="modify_form_visible" width="40%">
        <el-form :model="changeForm" status-icon :rules="keyRules" ref="changeForm" label-width="120px">
            <el-form-item label="请输入旧密码" prop="old_passwd">
                <el-input v-model="changeForm.old_passwd" type="password" autocomplete="off"></el-input>
            </el-form-item>
            <el-form-item label="请输入新密码" prop="new_passwd">
                <el-input v-model="changeForm.new_passwd" type="password" autocomplete="off"></el-input>
            </el-form-item>
            <el-form-item label="再次输入新密码" prop="confirm_new_passwd">
                <el-input v-model="changeForm.confirm_new_passwd" type="password" autocomplete="off"></el-input>
            </el-form-item>
        </el-form>
        <div class="dialog-footer">
            <el-button @click="modify_form_visible = false">取 消</el-button>
            <el-button type="primary" @click="submitForm('changeForm')">提 交</el-button>
        </div>
    </el-dialog>
</template>

<script>
    // eslint-disable-next-line no-unused-vars
    import Menu from './Menu'
    export default {
        name: 'Person',
        data () {
            var confirmNewPasswd = (rule, value, callback) => {
                if (value !== this.changeForm.new_passwd) {
                    callback(new Error("两次密码输入不一致！"))
                } else {
                    callback()
                }
            }
            return {
                username: "lyh",
                password: "",
                real_name: "林伊菡",
                gender: "女",
                age: "21",
                modify_form_visible: false,
                changeForm: {
                    old_passwd: "",
                    new_passwd: "",
                    confirm_new_passwd: "",
                },
                keyRules: {
                    confirm_new_passwd: [
                        {
                            validator: confirmNewPasswd,
                            trigger: 'blur'
                        }
                    ]
                }
            }
        },
        methods: {
            // eslint-disable-next-line no-unused-vars
            submitForm (form) {
                if (this.changeForm.old_passwd !== "123") {
                    this.$message.error("旧密码错误！")
                } else {
                    this.$message.success("密码修改成功")
                    this.modify_form_visible = false
                }
            }
        },
    }
</script>

<style scoped>
    .info_row {
        position: relative;
    }
    .info_key {
        position: relative;
        display: inline-block;
        text-align: right;
        width: 100px;
        font-size: 20px;
    }
    .info_box {
        position: relative;
        display: inline-block;
        text-align: left;
        height: 20px;
        line-height: 20px;
        width: 350px;
        font-size: 20px;
        border: solid;
        border-color: gainsboro;
        border-radius: 5px;
        border-width: 2px;
        margin-top: 30px;
        margin-left: 30px;
        margin-right: 30px;
        padding: 10px 20px
    }
    .modify_button {
        position: relative;
        width: 80px;
        vertical-align: middle;
        height: 42px;
        margin-bottom: 8px;
    }
    ::v-deep .info_box .el-input__inner {
        color: black;
    }
</style>