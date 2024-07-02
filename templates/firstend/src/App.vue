<template>
    <div id="app">
      <Layout></Layout>
      <router-view></router-view>
    </div>
</template>

<script>
import Layout from "@/components/Header.vue";
export default {
    name: "App",
    components: {Layout},
}
</script>
<style>
body {
    height: 100vh;
    margin: 0; /* 重置默认边距 */
    background-image: linear-gradient(30deg, #ef008f, #6ec3f4, #7038ff, #ffba27);
    font-family: "Andika New Basic", sans-serif;
    perspective: 1500px;
}

#app {
    height: 100%; /* 使 #app 高度占满整个视口 */
}
</style>
