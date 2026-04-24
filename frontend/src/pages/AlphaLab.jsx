import { useNavigate } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { 
  Row, 
  Col, 
  Card, 
  Table, 
  Tag, 
  Space, 
  Typography,
  Select,
  Input,
  Button,
  message,
} from 'antd'
import {
  ExperimentOutlined,
  LikeOutlined,
  DislikeOutlined,
  SyncOutlined,
} from '@ant-design/icons'
import { useState } from 'react'
import api from '../services/api'

const { Title, Text } = Typography
const { Search } = Input

export default function AlphaLab() {
  const navigate = useNavigate()
  const [pagination, setPagination] = useState({
    current: 1,
    pageSize: 20,
    total: 0,
  })
  const [syncing, setSyncing] = useState(false)

  // Fetch alphas
  const { data, isLoading, refetch } = useQuery({
    queryKey: ['alphas', pagination.current, pagination.pageSize],
    queryFn: () => api.getAlphas({ 
      limit: pagination.pageSize, 
      offset: (pagination.current - 1) * pagination.pageSize 
    }),
    keepPreviousData: true,
  })
  
  // Handle both Array (Old) and Object (New) response formats
  let alphas = []
  let total = 0
  
  if (Array.isArray(data)) {
    alphas = data
    total = data.length
  } else if (data && data.items && typeof data.total === 'number') {
    alphas = data.items
    total = data.total
  }
  


  const handleTableChange = (newPagination) => {
    setPagination(prev => ({
      ...prev,
      current: newPagination.current,
      pageSize: newPagination.pageSize
    }))
  }

  const handleSync = async () => {
    setSyncing(true)
    try {
      const res = await api.syncAlphas()
      message.success(`Sync started: ${res.message}`)
      setTimeout(() => refetch(), 2000)
    } catch (error) {
      message.error('Sync failed: ' + error.message)
    } finally {
      setSyncing(false)
    }
  }

  const columns = [
    {
      title: 'Name',
      dataIndex: 'name',
      key: 'name',
      width: 200,
      render: (text, record) => (
        <Space direction="vertical" size={0}>
          <Text strong>{text || 'anonymous'}</Text>
          <Text type="secondary" style={{ fontSize: 11, fontFamily: 'monospace' }}>
            {record.expression}
          </Text>
        </Space>
      )
    },
    {
      title: 'Type',
      dataIndex: 'type',
      key: 'type',
      width: 100,
    },
    {
      title: 'Date Created',
      dataIndex: 'created_at',
      key: 'created_at',
      width: 120,
      render: (text) => text ? new Date(text).toLocaleDateString() : '-'
    },
    {
      title: 'Region',
      dataIndex: 'region',
      key: 'region',
      width: 80,
    },
    {
      title: 'Quality',
      dataIndex: 'quality_status',
      key: 'quality_status',
      width: 110,
      render: (s) => {
        const status = s || 'PENDING'
        const color =
          status === 'PASS'
            ? 'green'
            : (status === 'PROMISING'
              ? 'blue'
              : (status === 'OPTIMIZE'
                ? 'gold'
                : (status === 'FAIL' ? 'red' : 'default')))
        return <Tag color={color}>{status}</Tag>
      },
    },
    {
      title: 'Sharpe',
      dataIndex: 'sharpe',
      key: 'sharpe',
      width: 100,
      render: (val) => val ? val.toFixed(2) : '-'
    },
    {
      title: 'Returns',
      dataIndex: 'returns',
      key: 'returns',
      width: 100,
      render: (val) => val ? `${(val * 100).toFixed(2)}%` : '-'
    },
    {
      title: 'Turnover',
      dataIndex: 'turnover',
      key: 'turnover',
      width: 100,
      render: (val) => val ? `${(val * 100).toFixed(2)}%` : '-'
    },
    {
      title: 'Drawdown',
      dataIndex: 'drawdown',
      key: 'drawdown',
      width: 100,
      render: (val) => val ? `${(val * 100).toFixed(2)}%` : '-'
    },
    {
      title: 'Margin',
      dataIndex: 'margin',
      key: 'margin',
      width: 100,
      render: (val) => val ? `${(val * 10000).toFixed(2)}‱` : '-'
    },
    {
      title: 'Fitness',
      dataIndex: 'fitness',
      key: 'fitness',
      width: 100,
      render: (val) => val ? val.toFixed(2) : '-'
    },
  ]

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 24 }}>
        <Title level={3} style={{ margin: 0 }}>
          <ExperimentOutlined style={{ marginRight: 12, color: '#00d4ff' }} />
          Alpha Lab
        </Title>
        <Button 
          type="primary" 
          icon={<SyncOutlined spin={syncing} />}
          onClick={handleSync}
          loading={syncing}
        >
          Sync Alphas
        </Button>
      </div>

      {/* Filters */}
      <Card className="glass-card" style={{ marginBottom: 16 }}>
        <Row gutter={16}>
          <Col xs={24} sm={8} lg={6}>
            <Select
              placeholder="Filter by Quality"
              style={{ width: '100%' }}
              allowClear
              options={[
                { value: 'PASS', label: 'PASS' },
                { value: 'PROMISING', label: 'PROMISING' },
                { value: 'OPTIMIZE', label: 'OPTIMIZE' },
                { value: 'FAIL', label: 'FAIL' },
                { value: 'REJECT', label: 'REJECT' },
                { value: 'PENDING', label: 'PENDING' },
              ]}
            />
          </Col>
          <Col xs={24} sm={8} lg={6}>
            <Select
              placeholder="Filter by Region"
              style={{ width: '100%' }}
              allowClear
              options={[
                { value: 'USA', label: 'USA' },
                { value: 'CHN', label: 'China' },
                { value: 'ASI', label: 'Asia' },
                { value: 'EUR', label: 'Europe' },
                { value: 'GLB', label: 'Global' },
                { value: 'HKG', label: 'Hong Kong' },
                { value: 'JPN', label: 'Japan' },
                { value: 'KOR', label: 'Korea' },
                { value: 'TWN', label: 'Taiwan' },
                { value: 'VNM', label: 'Vietnam' },
                { value: 'THA', label: 'Thailand' },
                { value: 'IND', label: 'India' },
                { value: 'AMR', label: 'America' },
              ]}
            />
          </Col>
          <Col xs={24} sm={8} lg={6}>
            <Select
              placeholder="Filter by Feedback"
              style={{ width: '100%' }}
              allowClear
              options={[
                { value: 'LIKED', label: '👍 Liked' },
                { value: 'DISLIKED', label: '👎 Disliked' },
                { value: 'NONE', label: 'Not Rated' },
              ]}
            />
          </Col>
          <Col xs={24} sm={24} lg={6}>
            <Search placeholder="Search expression..." />
          </Col>
        </Row>
      </Card>

      {/* Alpha Table */}
      <Card className="glass-card">
      <Table 
          columns={columns} 
          dataSource={alphas}
          rowKey="id"
          loading={isLoading}
          size="small"
          pagination={{
            ...pagination,
            total: total,
            showSizeChanger: true,
            showTotal: (total) => `Total ${total} items`,
          }}
          onChange={handleTableChange}
          title={() => <Text strong>Total Alphas: {total}</Text>}
          onRow={(record) => ({
            onClick: () => navigate(`/alphas/${record.id}`),
            style: { cursor: 'pointer' },
          })}
        />
      </Card>
    </div>
  )
}
