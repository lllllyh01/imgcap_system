<template>
    <el-table :data="tableData" style="width: 100%">
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
          <el-popover effect="light" placement="top" width="auto">
            <template #default>
              <div>name: {{ scope.row.name }}</div>
              <div>address: {{ scope.row.address }}</div>
            </template>
            <template #reference>
              <el-tag>{{ scope.row.name }}</el-tag>
            </template>
          </el-popover>
        </template>
      </el-table-column>
      <el-table-column label="描述" width="500">
        <template #default="scope">
          <span style="font-size: 15px;">{{ scope.row.address }}</span>
        </template>
      </el-table-column>
      <el-table-column label="描述不准确？">
        <template #default="scope">
          <el-button  @click="handleEdit(scope.$index, scope.row)" style="font-size: 15px;"
            >重新生成</el-button
          >
        </template>
      </el-table-column>
    </el-table>
  </template>

  <script lang="ts" setup>
    import { Timer } from '@element-plus/icons-vue'
    
    interface User {
      date: string
      name: string
      address: string
    }
    
    const handleEdit = (index: number, row: User) => {
      console.log(index, row)
    }

    const tableData: User[] = [
      {
        date: '2016-05-03',
        name: 'Tom',
        address: 'No. 189, Grove St, Los Angeles',
      },
      {
        date: '2016-05-02',
        name: 'Tom',
        address: 'No. 189, Grove St, Los Angeles',
      },
      {
        date: '2016-05-04',
        name: 'Tom',
        address: 'No. 189, Grove St, Los Angeles',
      },
      {
        date: '2016-05-01',
        name: 'Tom',
        address: 'No. 189, Grove St, Los Angeles',
      },
    ]
    </script>