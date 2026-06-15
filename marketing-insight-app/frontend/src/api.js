import axios from 'axios'

const api = axios.create({
  baseURL: '/api',
  timeout: 120000
})

export const getHealth = () => api.get('/health')
export const getCategories = () => api.get('/categories')
export const getInsights = (payload) => api.post('/insights', payload)
export const runCluster = (payload) => api.post('/cluster', payload)
export const askAssistant = (payload) => api.post('/chat', payload)
