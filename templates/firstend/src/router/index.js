import Vue from 'vue'
import Router from 'vue-router'

import Header from "@/components/Header.vue";
import prepic from "@/components/prepic.vue";
import BasicTask from "@/components/BasicTask.vue";
import StyleTransfer from "@/components/StyleTransfer.vue";
import Colorizer from "@/components/Colorizer.vue";
import HomePage from "@/components/HomePage.vue";
Vue.use(Router)

const routes = [
    {
        path: '/',
        redirect:'/home',
        component: Header,
        hidden: true,
    },
    {
      path:'/home',
      component: HomePage,
      hidden: true,
    },
    {
      path:'/basic',
      component: BasicTask,
      hidden: true
    },
    {
      path:'/restoration',
      component: Colorizer,
      hidden: true
    },
    {
      path:'/transfer',
      component: StyleTransfer,
      hidden: true
    },
];

const router = new Router({
    mode: 'history',//取消哈希模式
    routes
});

export default router;
