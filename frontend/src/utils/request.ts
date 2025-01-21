import axios from 'axios'
import { ElMessage } from 'element-plus'

const request = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL,
  timeout: 5000,
  withCredentials: true
})

// 请求拦截器
request.interceptors.request.use(
  (config) => {
    const userCookie = localStorage.getItem('userCookie')
    if (userCookie) {
      const cookieObj = JSON.parse(userCookie)
      const cookieString = Object.entries(cookieObj)
        .map(([key, value]) => `${key}=${value}`)
        .join('; ')
      config.headers.Cookie = cookieString
    }
    return config
  },
  (error) => {
    return Promise.reject(error)
  }
)

// 响应拦截器
request.interceptors.response.use(
  (response) => {
    // 直接返回响应数据，不做额外处理
    return response
  },
  (error) => {
    ElMessage.error(error.response?.data?.message || '请求失败')
    return Promise.reject(error)
  }
)

export default request
