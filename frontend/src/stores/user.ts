import { defineStore } from 'pinia'
import { ref } from 'vue'
import request from '@/utils/request'

interface Cookie {
  sessionid: string
  jgbsessid: string
  path: string
  domain: string
  THSFT_USERID: string
  u_name: string
  userid: string
  user: string
  ticket: string
  escapename: string
  version: string
  securities: string
  platform: string
  ftuser_pf: string
}

interface UserState {
  cookie: Cookie | null
  isLoggedIn: boolean
  username: string
}

export const useUserStore = defineStore('user', () => {
  const state = ref<UserState>({
    cookie: null,
    isLoggedIn: false,
    username: ''
  })

  const login = async (username: string, password: string) => {
    try {
      const response = await request.post('/login', { username, password })

      // 检查响应状态和数据
      if (response.status === 200 && response.data.cookie) {
        state.value.cookie = response.data.cookie
        state.value.isLoggedIn = true
        state.value.username = response.data.cookie.u_name || username

        // 保存 cookie 到 localStorage
        localStorage.setItem('userCookie', JSON.stringify(response.data.cookie))
        return response.data
      } else {
        throw new Error('登录响应格式不正确')
      }
    } catch (error) {
      console.error('Login error:', error)
      throw error
    }
  }

  const logout = () => {
    state.value.cookie = null
    state.value.isLoggedIn = false
    state.value.username = ''
    localStorage.removeItem('userCookie')
  }

  // 初始化状态（从 localStorage 恢复）
  const initState = () => {
    const savedCookie = localStorage.getItem('userCookie')
    if (savedCookie) {
      state.value.cookie = JSON.parse(savedCookie)
      state.value.isLoggedIn = true
      state.value.username = state.value.cookie?.u_name || ''
    }
  }

  return {
    state,
    login,
    logout,
    initState
  }
})
