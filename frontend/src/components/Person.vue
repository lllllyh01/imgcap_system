<template>
    <Menu></Menu>
    <div class="info_row">
        <span class="info_key">用户名：</span>
        <span class="info_box">{{username}}</span>
        <el-button class="modify_button" type="primary" @click="modify_form_visible = true">修改</el-button>
    </div>
    <div class="info_row">
        <span class="info_key">密码：</span>
        <span class="info_box">*****</span>
        <el-button class="modify_button" type="primary" @click="modify_form_visible = true">修改</el-button>
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
    <el-dialog title="修改用户名" :visible="modify_form_visible" width="40%">
        <el-form :model="changeForm" status-icon :rules="keyRules" ref="changeForm" label-width="100px">
            <p style="margin:0 0 10px 100px">原用户名：{{username}}</p>
            <el-form-item label="新用户名" prop="new_username">
            <el-input v-model="changeForm.new_username" autocomplete="off"></el-input>
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
            return {
                username: "lyh",
                password: "",
                real_name: "林伊菡",
                gender: "女",
                age: "21",
                modify_form_visible: false,
            }
        },
        methods: {
            modify_username() {
                this.disable_un = false;
                console.log("in modify_username");
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