<template>
  <div class="min-h-screen bg-gray-100 flex items-center justify-center py-12 px-4 sm:px-6 lg:px-8">
    <div class="max-w-md w-full space-y-8 bg-white p-8 rounded-lg shadow-lg">
      <div>
        <h2 class="mt-6 text-center text-3xl font-extrabold text-gray-900">
          系统登录
        </h2>
      </div>

      <el-form
        :model="loginForm"
        :rules="rules"
        ref="loginFormRef"
        class="mt-8 space-y-6"
      >
        <el-form-item prop="username" class="rounded-md shadow-sm -space-y-px">
          <el-input
            v-model="loginForm.username"
            placeholder="请输入用户名"
            :prefix-icon="User"
            class="appearance-none rounded-none relative block w-full px-3 py-2 border border-gray-300 placeholder-gray-500 text-gray-900 rounded-t-md focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 focus:z-10 sm:text-sm"
          />
        </el-form-item>

        <el-form-item prop="password">
          <el-input
            v-model="loginForm.password"
            type="password"
            placeholder="请输入密码"
            :prefix-icon="Lock"
            show-password
            class="appearance-none rounded-none relative block w-full px-3 py-2 border border-gray-300 placeholder-gray-500 text-gray-900 rounded-b-md focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 focus:z-10 sm:text-sm"
          />
        </el-form-item>

        <div class="flex items-center justify-between">
          <el-checkbox v-model="rememberMe" class="text-sm text-gray-600">
            记住我
          </el-checkbox>

          <div class="text-sm">
            <a href="#" class="font-medium text-indigo-600 hover:text-indigo-500">
              忘记密码？
            </a>
          </div>
        </div>

        <el-form-item>
          <el-button
            type="primary"
            @click="handleLogin"
            :loading="loading"
            class="w-full flex justify-center py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
          >
            {{ loading ? '登录中...' : '登录' }}
          </el-button>
        </el-form-item>
      </el-form>
    </div>
  </div>
</template>

<script lang="ts" setup>
import { ref, reactive, onMounted } from 'vue'
import type { FormInstance, FormRules } from 'element-plus'
import { ElMessage } from 'element-plus'
import { User, Lock } from '@element-plus/icons-vue'
import { useRouter } from 'vue-router'
import { useUserStore } from '@/stores/user'

const router = useRouter()
const userStore = useUserStore()

interface LoginForm {
  username: string
  password: string
}

// 表单引用
const loginFormRef = ref<FormInstance>()

// 表单数据
const loginForm = reactive<LoginForm>({
  username: '',
  password: ''
})

// 记住我选项
const rememberMe = ref(false)

// 加载状态
const loading = ref(false)

// 表单验证规则
const rules = reactive<FormRules>({
  username: [
    { required: true, message: '请输入用户名', trigger: 'blur' },
    { min: 3, max: 20, message: '长度在 3 到 20 个字符', trigger: 'blur' }
  ],
  password: [
    { required: true, message: '请输入密码', trigger: 'blur' },
    { min: 6, max: 20, message: '长度在 6 到 20 个字符', trigger: 'blur' }
  ]
})

// 组件挂载时检查是否有保存的用户名
onMounted(() => {
  const savedUsername = localStorage.getItem('rememberedUsername')
  if (savedUsername) {
    loginForm.username = savedUsername
    rememberMe.value = true
  }
})

// 登录处理函数
const handleLogin = async () => {
  if (!loginFormRef.value) return

  try {
    await loginFormRef.value.validate()
    loading.value = true

    const result = await userStore.login(loginForm.username, loginForm.password)

    // 检查登录结果
    if (result && result.cookie) {
      // 处理记住用户名
      if (rememberMe.value) {
        localStorage.setItem('rememberedUsername', loginForm.username)
      } else {
        localStorage.removeItem('rememberedUsername')
      }

      ElMessage.success('登录成功')
      router.push('/')
    } else {
      throw new Error('登录响应数据不完整')
    }
  } catch (error) {
    console.error('登录失败:', error)
    // 根据具体错误类型显示不同的错误信息
    if (error.response && error.response.data) {
      ElMessage.error(error.response.data.message || '登录失败，请检查用户名和密码')
    } else {
      ElMessage.error('登录失败，请稍后重试')
    }
  } finally {
    loading.value = false
  }
}
</script>

<style scoped>
:deep(.el-input__wrapper) {
  box-shadow: none !important;
}

:deep(.el-input__inner) {
  background: transparent;
}
</style>
