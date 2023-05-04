<template>
    <h3>用户名：lyh</h3>
    <el-button class="modify" @click="modify_username">修改</el-button>
</template>

<script>
    export default {
        name: 'Person',
        data () {
            return {
                username: "",
                passwd: "",
                gender: "",
                age: ""
            }
        },
        methods: {
            modify_username() {
                console.log("in modify_username");
            }
        },
    }
</script>