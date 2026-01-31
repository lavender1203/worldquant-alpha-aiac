import axios from 'axios'

const API_BASE = '/api/v1'

const client = axios.create({
  baseURL: API_BASE,
  headers: {
    'Content-Type': 'application/json',
  },
})

// API functions
const api = {
  // Datasets & Fields
  getDatasets: async (params = {}) => {
    const { data } = await client.get('/datasets', { params })
    return data
  },

  getDataset: async (id) => {
    const { data } = await client.get(`/datasets/${id}`)
    return data
  },

  syncDatasets: async (region, universe) => {
    const { data } = await client.post('/datasets/sync', null, { params: { region, universe } })
    return data
  },

  getDatasetCategories: async () => {
    const { data } = await client.get('/datasets/categories')
    return data
  },

  getDatasetFields: async (datasetId, params = {}) => {
    const { data } = await client.get(`/datasets/${datasetId}/fields`, { params })
    return data
  },

  syncDatasetFields: async (datasetId, region, universe) => {
    const { data } = await client.post(`/datasets/${datasetId}/sync-fields`, null, { 
      params: { region, universe } 
    })
    return data
  },

  // Operators
  getOperators: async (params = {}) => {
    const { data } = await client.get('/operators', { params })
    return data
  },

  syncOperators: async () => {
    const { data } = await client.post('/operators/sync')
    return data
  },

  // Dashboard / Stats
  getDailyStats: async (date) => {
    const params = date ? { date } : {}
    const { data } = await client.get('/stats/daily', { params })
    return data
  },

  getKPIMetrics: async () => {
    const { data } = await client.get('/stats/kpi')
    return data
  },

  getActiveTasks: async () => {
    const { data } = await client.get('/stats/active-tasks')
    return data
  },

  // Tasks
  getTasks: async (params = {}) => {
    const { data } = await client.get('/tasks', { params })
    return data
  },

  getTask: async (id) => {
    const { data } = await client.get(`/tasks/${id}`)
    return data
  },

  getTaskTrace: async (id) => {
    const { data } = await client.get(`/tasks/${id}/trace`)
    return data
  },

  createTask: async (taskData) => {
    const { data } = await client.post('/tasks', taskData)
    return data
  },

  startTask: async (id) => {
    const { data } = await client.post(`/tasks/${id}/start`)
    return data
  },

  interveneTask: async (id, action, parameters = {}) => {
    const { data } = await client.post(`/tasks/${id}/intervene`, { action, parameters })
    return data
  },

  getTaskRuns: async (taskId) => {
    const { data } = await client.get(`/tasks/${taskId}/runs`)
    return data
  },

  getRun: async (runId) => {
    const { data } = await client.get(`/runs/${runId}`)
    return data
  },

  getRunTrace: async (runId) => {
    const { data } = await client.get(`/runs/${runId}/trace`)
    return data
  },

  getRunAlphas: async (runId, params = {}) => {
    const { data } = await client.get(`/runs/${runId}/alphas`, { params })
    return data
  },

  // Alphas
  getAlphas: async (params = {}) => {
    const { data } = await client.get('/alphas', { params })
    return data
  },

  getAlpha: async (id) => {
    const { data } = await client.get(`/alphas/${id}`)
    return data
  },

  getAlphaTrace: async (id) => {
    const { data } = await client.get(`/alphas/${id}/trace`)
    return data
  },

  submitAlphaFeedback: async (id, rating, comment = null) => {
    const { data } = await client.post(`/alphas/${id}/feedback`, { rating, comment })
    return data
  },

  syncAlphas: async () => {
    const { data } = await client.post('/alphas/sync')
    return data
  },

  // Knowledge
  getKnowledgeEntries: async (params = {}) => {
    const { data } = await client.get('/knowledge', { params })
    return data
  },

  getSuccessPatterns: async (limit = 20) => {
    const { data } = await client.get('/knowledge/success-patterns', { params: { limit } })
    return data
  },

  getFailurePitfalls: async (limit = 50) => {
    const { data } = await client.get('/knowledge/failure-pitfalls', { params: { limit } })
    return data
  },

  createKnowledgeEntry: async (entryData) => {
    const { data } = await client.post('/knowledge', entryData)
    return data
  },

  updateKnowledgeEntry: async (id, updates) => {
    const { data } = await client.put(`/knowledge/${id}`, updates)
    return data
  },

  deleteKnowledgeEntry: async (id) => {
    const { data } = await client.delete(`/knowledge/${id}`)
    return data
  },

  // Knowledge Base Management
  getKnowledgeStats: async () => {
    const { data } = await client.get('/knowledge/stats')
    return data
  },

  getExternalKnowledge: async (source = null) => {
    const params = source ? { source } : {}
    const { data } = await client.get('/knowledge/categories/external', { params })
    return data
  },

  getSystemKnowledge: async (entryType = null) => {
    const params = entryType ? { entry_type: entryType } : {}
    const { data } = await client.get('/knowledge/categories/system', { params })
    return data
  },

  createKnowledge: async (entryData) => {
    const { source_type, source_url, ...rest } = entryData
    const params = { source_type }
    if (source_url) params.source_url = source_url
    const { data } = await client.post('/knowledge/external', rest, { params })
    return data
  },

  updateKnowledge: async (id, updates) => {
    const { data } = await client.put(`/knowledge/${id}`, updates)
    return data
  },

  deleteKnowledge: async (id) => {
    const { data } = await client.delete(`/knowledge/${id}`)
    return data
  },

  // Forum
  syncForum: async (options = {}) => {
    const { data } = await client.post('/knowledge/sync/forum', options)
    return data
  },

  searchForum: async (query, limit = 10) => {
    const { data } = await client.get('/knowledge/forum/search', { params: { query, limit } })
    return data
  },

  importForumPost: async (post) => {
    const { data } = await client.post('/knowledge/forum/import', post)
    return data
  },

  // Papers
  getPapers: async () => {
    const { data } = await client.get('/knowledge/papers')
    return data
  },

  downloadPaper: async (paperData) => {
    const { data } = await client.post('/knowledge/papers/download', paperData)
    return data
  },

  // Config
  getConfig: async () => {
    const { data } = await client.get('/config')
    return data
  },

  updateThresholds: async (thresholds) => {
    const { data } = await client.put('/config/thresholds', thresholds)
    return data
  },

  // Credentials Management
  getCredentialsStatus: async () => {
    const { data } = await client.get('/config/credentials')
    return data
  },

  setBrainCredentials: async (email, password) => {
    const { data } = await client.post('/config/credentials/brain', { email, password })
    return data
  },

  setLLMCredentials: async (apiKey, baseUrl, model) => {
    const { data } = await client.post('/config/credentials/llm', { 
      api_key: apiKey, 
      base_url: baseUrl, 
      model 
    })
    return data
  },

  testBrainCredentials: async () => {
    const { data } = await client.post('/config/credentials/brain/test')
    return data
  },

  deleteCredential: async (key) => {
    const { data } = await client.delete(`/config/credentials/${key}`)
    return data
  },

  // Priority Datasets
  getAllPriorityDatasets: async () => {
    const { data } = await client.get('/config/priority-datasets')
    return data
  },

  getPriorityDatasets: async (region) => {
    const { data } = await client.get(`/config/priority-datasets/${region}`)
    return data
  },

  setPriorityDatasets: async (region, datasetIds) => {
    const { data } = await client.put(`/config/priority-datasets/${region}`, {
      region,
      dataset_ids: datasetIds
    })
    return data
  },

  deletePriorityDatasets: async (region) => {
    const { data } = await client.delete(`/config/priority-datasets/${region}`)
    return data
  },
}

export default api
