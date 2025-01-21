import './assets/main.css'

import { createApp } from 'vue'
import { createPinia } from 'pinia'
import './index.css'

import ElementPlus from 'element-plus'
import 'element-plus/dist/index.css'

import App from './App.vue'
import router from './router'
import { useUserStore } from '@/stores/user.ts'

const app = createApp(App)

app.use(createPinia())

const userStore = useUserStore()
userStore.initState()

app.use(router)
app.use(ElementPlus)

app.mount('#app')
