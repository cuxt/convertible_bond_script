<template>
  <div class="app-container">
    <AppHeader />
    <main class="main-content">
      <router-view></router-view>
    </main>
    <AppFooter />
  </div>
</template>

<script lang="ts" setup>
import AppHeader from './components/AppHeader.vue'
import AppFooter from './components/AppFooter.vue'
</script>

<style>
/* 重置基础样式 */
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

html, body {
  width: 100vw;
  min-height: 100vh;
  margin: 0;
  padding: 0;
  overflow-x: hidden;
}

#app {
  width: 100vw;
  min-height: 100vh;
  display: flex;
  flex-direction: column;
  padding: 0;
}

.app-container {
  width: 100vw;
  min-height: 100vh;
  display: flex;
  flex-direction: column;
}

.main-content {
  flex: 1;
  width: 100%;
}
</style>
