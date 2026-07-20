import './assets/main.css'
import 'mathjax/es5/tex-mml-chtml.js'

import { createApp } from 'vue'
import App from './App.vue'
import { mathjaxDirective } from './directives/mathjax'

const app = createApp(App)

app.directive('mathjax', mathjaxDirective)

app.mount('#app')
