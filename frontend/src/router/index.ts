import { createRouter, createWebHistory } from 'vue-router'
import type { RouteRecordRaw } from 'vue-router'
import HomeView from '../views/HomeView.vue'
import Login from '../components/Login.vue'
import DescriptionGeneration from '../components/DescriptionGeneration.vue'
import History from '../components/History.vue'
import Menu from '../components/Menu.vue'
import Person from '../components/Person.vue'

const routes: RouteRecordRaw[] = [
  {
    path: '/',
    name: 'home',
    component: HomeView
  },
  // {
  //   path: '/about',
  //   name: 'about',
  //   // route level code-splitting
  //   // this generates a separate chunk (about.[hash].js) for this route
  //   // which is lazy-loaded when the route is visited.
  //   component: () => import(/* webpackChunkName: "about" */ '../views/AboutView.vue')
  // },
  // {
  //   path: '/description_generation',
  //   name: 'DescriptionGeneration',
  //   component: DescriptionGeneration
  // },
  {
    path: '/login',
    name: 'Login',
    component: Login
  },
  {
    path: '/menu',
    name: 'Menu',
    component: Menu,
    children: [
      {path: '/description_generation', name: 'DescriptionGeneration', component: DescriptionGeneration, meta: {title: ['生成描述']}},
      {path: '/person', name: 'Person', component: Person, meta: {title: ['个人信息']}},
      {path: '/history', name: 'History', component: History, meta: {title: ['历史记录']}}
    ]
  }
]

const router = createRouter({
  history: createWebHistory(process.env.BASE_URL),
  routes
})

export default router
