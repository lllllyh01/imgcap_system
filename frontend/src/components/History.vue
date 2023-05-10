<template>
  <el-table :data="historys"
  style="width: 100%"
  :header-cell-style="{fontSize: '25px'}">
    <el-table-column label="时间" width="300">
      <template #default="scope">
        <div style="display: flex; align-items: center">
          <i class="el-icon-timer"></i>
          <span style="margin-left: 10px; font-size: 20px;">{{ scope.row.date }}</span>
        </div>
      </template>
    </el-table-column>
    <el-table-column label="图像" width="300">
      <template #default="scope">
        <el-image style="width: 200px; height: 200px;" :src="scope.row.img_url" alt="FAILED" fit="contain"/>
      </template>
    </el-table-column>
    <el-table-column label="描述" width="500">
      <template #default="scope">
        <span style="display: block; font-size: 15px; margin-right: 15px;">{{ scope.row.description }}</span>
      </template>
    </el-table-column>
    <el-table-column label="操作">
      <template #default="scope">
        <el-button type="danger"  @click="deleteHistory(scope.$index, scope.row)" style="font-size: 15px;"
          >删除</el-button
        >
      </template>
    </el-table-column>
  </el-table>
</template>

<script>
  import { ElMessage, ElMessageBox } from 'element-plus'
  export default {
    name: 'History',
    data () {
      return {
        historys: [
          {
            date: '2023-05-03',
            img_url: require('../../../model_vscode/CXR1660_IM-0436-2002.png'),
            description: 'No active disease. Both lungs are clear and expanded. Heart and mediastinum within normal limits. Degenerative changes in the thoracic spine.',
          },
        ]
      }
    },
    methods: {
      // eslint-disable-next-line no-unused-vars
      deleteHistory (index, row) {
        ElMessageBox.confirm('删除后，此条记录不可恢复。是否确认删除？','提示',
        {
          confirmButtonText: '确认',
          cancelButtonText: '取消',
          type: 'warning',
        })
        .then(() => {
          this.historys.splice(0, 1);
          console.log(this.historys)
          ElMessage({
            type: 'success',
            message: '删除成功',
          })
        })
        .catch(() => {
          
        })
      }
    }
  }
</script>