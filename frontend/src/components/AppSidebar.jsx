import { useState } from 'react'
import { useNavigate, useLocation } from 'react-router-dom'
import { Layout, Menu } from 'antd'
import {
  DashboardOutlined,
  ThunderboltOutlined,
  ExperimentOutlined,
  SettingOutlined,
  RocketOutlined,
  DatabaseOutlined,
  ApiOutlined,
} from '@ant-design/icons'

const { Sider } = Layout

const menuItems = [
  {
    key: '/dashboard',
    icon: <DashboardOutlined />,
    label: '仪表盘',
  },
  {
    key: '/tasks',
    icon: <ThunderboltOutlined />,
    label: '任务管理',
  },
  {
    key: '/alphas',
    icon: <ExperimentOutlined />,
    label: '因子实验室',
  },
  {
    key: '/data',
    icon: <DatabaseOutlined />,
    label: '数据管理',
  },
  {
    key: '/mcp',
    icon: <ApiOutlined />,
    label: 'MCP 管理',
  },
  {
    key: '/config',
    icon: <SettingOutlined />,
    label: '配置中心',
  },
]

export default function AppSidebar() {
  const [collapsed, setCollapsed] = useState(false)
  const navigate = useNavigate()
  const location = useLocation()

  const handleMenuClick = ({ key }) => {
    navigate(key)
  }

  return (
    <Sider
      collapsible
      collapsed={collapsed}
      onCollapse={setCollapsed}
      style={{
        background: 'linear-gradient(180deg, #131a2b 0%, #0a0e17 100%)',
        borderRight: '1px solid rgba(255, 255, 255, 0.1)',
      }}
    >
      <div style={{
        height: 64,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        borderBottom: '1px solid rgba(255, 255, 255, 0.1)',
      }}>
        <RocketOutlined style={{ fontSize: 24, color: '#00d4ff' }} />
        {!collapsed && (
          <span style={{
            marginLeft: 12,
            fontSize: 18,
            fontWeight: 600,
            color: '#00d4ff',
          }}>
            AIAC 2.0
          </span>
        )}
      </div>
      <Menu
        theme="dark"
        mode="inline"
        selectedKeys={[location.pathname]}
        items={menuItems}
        onClick={handleMenuClick}
        style={{ background: 'transparent', borderRight: 'none' }}
      />
    </Sider>
  )
}
