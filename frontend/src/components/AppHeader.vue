<template>
  <header class="w-full bg-white shadow-md">
    <nav class="w-full px-4 sm:px-6 lg:px-8">
      <div class="flex justify-between items-center h-16">
        <!-- Logo -->
        <div class="flex-shrink-0 flex items-center">
          <img class="h-8 w-auto" src="@/assets/logo.svg" alt="Logo" />
          <span class="ml-2 text-xl font-bold text-gray-800">vFinD</span>
        </div>

        <!-- Navigation Links -->
        <div class="hidden sm:flex sm:space-x-8">
          <router-link
            v-for="item in navigationItems"
            :key="item.name"
            :to="item.path"
            class="text-gray-600 hover:text-gray-900 px-3 py-2 rounded-md text-sm font-medium"
            :class="{ 'text-indigo-600': isCurrentPath(item.path) }"
          >
            {{ item.name }}
          </router-link>
        </div>

        <!-- User Menu -->
        <div class="flex items-center space-x-4">
          <template v-if="userStore.state.isLoggedIn">
            <el-dropdown>
              <span class="flex items-center cursor-pointer">
                <img
                  class="h-8 w-8 rounded-full"
                  src="https://avator.bxin.top/api/face"
                  alt="User avatar"
                />
                <span class="ml-2 text-sm font-medium text-gray-700">
                  {{ userStore.state.username }}
                </span>
              </span>
              <template #dropdown>
                <el-dropdown-menu>
                  <el-dropdown-item @click="navigateTo('/profile')">个人信息</el-dropdown-item>
                  <el-dropdown-item @click="navigateTo('/settings')">设置</el-dropdown-item>
                  <el-dropdown-item divided @click="handleLogout">退出登录</el-dropdown-item>
                </el-dropdown-menu>
              </template>
            </el-dropdown>
          </template>
          <template v-else>
            <router-link
              to="/login"
              class="text-gray-600 hover:text-gray-900 px-3 py-2 rounded-md text-sm font-medium"
            >
              登录
            </router-link>
          </template>
        </div>
      </div>
    </nav>
  </header>
</template>

<script lang="ts" setup>
import { ref, onMounted } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { useUserStore } from '@/stores/user'
import { ElMessage } from 'element-plus'

const router = useRouter()
const route = useRoute()
const userStore = useUserStore()

interface NavigationItem {
  name: string
  path: string
}

// 导航项
const navigationItems = ref<NavigationItem[]>([
  { name: '首页', path: '/' },
  { name: '产品', path: '/products' },
  { name: '服务', path: '/services' },
  { name: '关于我们', path: '/about' }
])

// 在组件挂载时初始化状态
onMounted(() => {
  userStore.initState()
})

// 判断当前路径
const isCurrentPath = (path: string): boolean => {
  return route.path === path
}

// 页面导航
const navigateTo = (path: string): void => {
  router.push(path)
}

// 退出登录
const handleLogout = async (): Promise<void> => {
  try {
    userStore.logout()
    ElMessage.success('退出登录成功')
    router.push('/login')
  } catch (error) {
    console.error('退出登录失败:', error)
    ElMessage.error('退出登录失败')
  }
}
</script>
