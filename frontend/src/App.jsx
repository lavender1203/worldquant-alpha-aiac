import { Routes, Route, Navigate } from 'react-router-dom'
import { Layout } from 'antd'
import AppSidebar from './components/AppSidebar'
import AppHeader from './components/AppHeader'
import Dashboard from './pages/Dashboard'
import TaskManagement from './pages/TaskManagement'
import TaskDetail from './pages/TaskDetail'
import AlphaLab from './pages/AlphaLab'
import AlphaDetail from './pages/AlphaDetail'
import ConfigCenter from './pages/ConfigCenter'
import DataManagement from './pages/DataManagement'
import MCPManagement from './pages/MCPManagement'

const { Content } = Layout

function App() {
  return (
    <Layout style={{ minHeight: '100vh' }}>
      <AppSidebar />
      <Layout>
        <AppHeader />
        <Content style={{ padding: '24px', overflow: 'auto' }}>
          <Routes>
            <Route path="/" element={<Navigate to="/dashboard" replace />} />
            <Route path="/dashboard" element={<Dashboard />} />
            <Route path="/tasks" element={<TaskManagement />} />
            <Route path="/tasks/:id" element={<TaskDetail />} />
            <Route path="/alphas" element={<AlphaLab />} />
            <Route path="/alphas/:id" element={<AlphaDetail />} />
            <Route path="/data" element={<DataManagement />} />
            <Route path="/mcp" element={<MCPManagement />} />
            <Route path="/config" element={<ConfigCenter />} />
          </Routes>
        </Content>
      </Layout>
    </Layout>
  )
}

export default App
