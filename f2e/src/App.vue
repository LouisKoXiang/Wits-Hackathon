<template>
  <router-view />
</template>

<script setup lang="ts">

console.log(import.meta.env.VITE_API_BASE_URL)
</script>

<style>
html, body {
  font-family: system-ui, sans-serif;
  margin: 0;
  padding: 0;
  background: #f9fafb;
}
</style>
