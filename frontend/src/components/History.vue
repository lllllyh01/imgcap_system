<template>
  <el-table :data="historys"
  style="width: 100%"
  :header-cell-style="{fontSize: '25px'}">
    <el-table-column label="时间" width="300">
      <template #default="scope">
        <div style="display: flex; align-items: center">
          <el-icon><timer /></el-icon>
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
    <el-table-column label="描述不准确？">
      <template #default="scope">
        <el-button  @click="goRegenerate(scope.$index, scope.row)" style="font-size: 15px;"
          >重新生成</el-button
        >
      </template>
    </el-table-column>
  </el-table>
</template>

<script>
  // eslint-disable-next-line no-unused-vars
  import { Timer } from '@element-plus/icons-vue'
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
      goRegenerate (index, row) {
        console.log(index, row)
      }
    }
  }
</script>