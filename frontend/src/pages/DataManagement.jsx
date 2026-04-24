import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { 
  Row, 
  Col, 
  Card, 
  Table, 
  Button, 
  Tag, 
  Space, 
  Typography,
  Input,
  Select,
  message,
  Tooltip,
  Tabs,
  Drawer,
  List,
  Descriptions
} from 'antd'
import {
  DatabaseOutlined,
  SyncOutlined,
  SearchOutlined,
  FunctionOutlined,
  UnorderedListOutlined,
  CheckCircleOutlined
} from '@ant-design/icons'
import api from '../services/api'

const { Title, Text, Paragraph } = Typography
const { Option } = Select

const REGION_UNIVERSE_MAP = {
  'USA': 'TOP3000',
  'CHN': 'TOP3000',
  'EUR': 'TOP2500',
  'ASI': 'MINVOL1M',
  'KOR': 'TOP600',
  'IND': 'TOP500'
}

// Component for Dataset Detail View (Separate Page Style)
const DatasetDetailView = ({ dataset, onBack }) => {
  const [activeTab, setActiveTab] = useState('fields')
  const [fieldSearch, setFieldSearch] = useState('')
  const [pagination, setPagination] = useState({ current: 1, pageSize: 20 })
  const queryClient = useQueryClient()

  // Fetch fields
  const { data: fieldsData, isLoading } = useQuery({
    queryKey: ['fields', dataset?.dataset_id, dataset?.region, pagination, fieldSearch],
    queryFn: () => {
        const universe = REGION_UNIVERSE_MAP[dataset.region] || dataset.universe || 'TOP3000'
        return api.getDatasetFields(dataset.dataset_id, { 
          region: dataset.region, 
          universe: universe,
          delay: dataset.delay || 1,
          search: fieldSearch,
          limit: pagination.pageSize,
          offset: (pagination.current - 1) * pagination.pageSize
        })
    },
    enabled: !!dataset,
    keepPreviousData: true,
  })

  // Sync fields mutation
  const syncFieldsMutation = useMutation({
    mutationFn: () => {
        const universe = REGION_UNIVERSE_MAP[dataset.region] || dataset.universe || 'TOP3000'
        return api.syncDatasetFields(dataset.dataset_id, dataset.region, universe)
    },
    onSuccess: (data) => {
      message.success(data.message)
      queryClient.invalidateQueries(['fields', dataset?.dataset_id])
    },
    onError: () => message.error('同步字段失败'),
  })

  const columns = [
    {
      title: '字段 (Field)',
      dataIndex: 'field_id',
      key: 'field_id',
      render: text => <Text code copyable>{text}</Text>,
      width: 250,
    },
    {
      title: '描述 (Description)',
      dataIndex: 'description',
      key: 'description',
      ellipsis: true,
    },
    {
      title: '类型 (Type)',
      dataIndex: 'field_type',
      key: 'field_type',
      width: 100,
      render: (text) => <Tag>{text || 'VECTOR'}</Tag>
    },
    {
      title: 'Pyramid Multiplier',
      dataIndex: 'pyramid_multiplier',
      key: 'pyramid_multiplier',
      width: 150,
      align: 'right',
      render: (val) => val?.toFixed(1)
    },
    {
      title: '覆盖率 (Coverage)',
      dataIndex: 'coverage',
      key: 'coverage',
      width: 120,
      align: 'right',
      render: (val) => val ? `${(val * 100).toFixed(0)}%` : '-'
    },
    {
      title: '日期覆盖 (Date Coverage)',
      dataIndex: 'date_coverage',
      key: 'date_coverage',
      width: 150,
      align: 'right',
      render: (val) => val ? `${(val * 100).toFixed(0)}%` : '-'
    },
    {
      title: 'Alphas',
      dataIndex: 'alpha_count',
      key: 'alpha_count',
      width: 80,
      align: 'right',
    }
  ]

  return (
    <div style={{ padding: 0 }}>
      {/* Header / Breadcrumb area */}
      <div style={{ marginBottom: 16 }}>
        <Button onClick={onBack} type="link" style={{ paddingLeft: 0 }}>
           &lt; 返回数据集列表
        </Button>
        <Title level={4} style={{ marginTop: 0 }}>
          {dataset.dataset_id} <Tag color="blue">{dataset.region}</Tag>
        </Title>
        <Paragraph type="secondary">{dataset.description}</Paragraph>
      </div>

      <div style={{ marginBottom: 16 }}>
         <Space>
            <Select defaultValue={dataset.region} disabled style={{ width: 80 }} />
            <Select defaultValue={dataset.delay || 1} disabled style={{ width: 60 }} />
            <Select defaultValue={dataset.universe || 'TOP3000'} disabled style={{ width: 120 }} />
         </Space>
      </div>

      <Tabs 
        activeKey={activeTab} 
        onChange={setActiveTab}
        type="card"
        items={[
            {
                key: 'fields',
                label: '字段列表 (Fields)',
                children: (
                    <>
                        <Row justify="space-between" style={{ marginBottom: 16 }}>
                            <Col>
                                <Input 
                                    placeholder="搜索字段名称、描述..." 
                                    prefix={<SearchOutlined />} 
                                    onChange={e => setFieldSearch(e.target.value)}
                                    style={{ width: 300 }}
                                    allowClear 
                                />
                            </Col>
                            <Col>
                                <Button 
                                    type="primary" 
                                    icon={<SyncOutlined spin={syncFieldsMutation.isLoading} />}
                                    loading={syncFieldsMutation.isLoading}
                                    onClick={() => syncFieldsMutation.mutate()}
                                >
                                    同步最新字段
                                </Button>
                            </Col>
                        </Row>
                        <Table
                            columns={columns}
                            dataSource={fieldsData?.results || []}
                            rowKey="field_id"
                            loading={isLoading}
                            pagination={{
                                ...pagination,
                                total: fieldsData?.total || 0,
                                showTotal: (total) => `共 ${total} 条`,
                                showSizeChanger: true,
                                onChange: (page, pageSize) => setPagination({ current: page, pageSize })
                            }}
                            size="middle"
                            bordered
                        />
                    </>
                )
            },
            {
                key: 'desc',
                label: '详细描述 (Description)',
                children: <Card><Paragraph>{dataset.description}</Paragraph></Card>
            }
        ]}
      />
    </div>
  )
}

// Operators Tab Component
const OperatorsTab = () => {
  const [search, setSearch] = useState('')
  const [category, setCategory] = useState(null)
  const queryClient = useQueryClient()

  const { data: operators, isLoading } = useQuery({
    queryKey: ['operators', search, category],
    queryFn: () => api.getOperators({ search, category }),
  })

  const syncMutation = useMutation({
    mutationFn: () => api.syncOperators(),
    onSuccess: (data) => {
      message.success(data.message)
      queryClient.invalidateQueries(['operators'])
    },
    onError: () => message.error('同步算子失败'),
  })

  const columns = [
    {
      title: '算子名称',
      dataIndex: 'name',
      key: 'name',
      width: 200,
      render: (text) => <Text strong code>{text}</Text>,
    },
    {
      title: '类别',
      dataIndex: 'category',
      key: 'category',
      width: 150,
      render: (text) => text ? <Tag color="cyan">{text}</Tag> : <Tag>General</Tag>,
    },
    {
      title: '描述',
      dataIndex: 'description',
      key: 'description',
      ellipsis: true,
    },
    {
      title: '状态',
      dataIndex: 'is_active',
      key: 'is_active',
      width: 100,
      render: (active) => active ? <Tag color="success">Active</Tag> : <Tag color="error">Inactive</Tag>
    }
  ]

  return (
    <div>
      <Row justify="space-between" style={{ marginBottom: 16 }}>
        <Col span={12}>
          <Space>
            <Input 
              placeholder="搜索算子..." 
              prefix={<SearchOutlined />} 
              onChange={e => setSearch(e.target.value)}
              style={{ width: 200 }}
              allowClear
            />
            <Select
              placeholder="类别筛选"
              style={{ width: 150 }}
              allowClear
              onChange={val => setCategory(val)}
            >
              <Option value="math">Math</Option>
              <Option value="time_series">Time Series</Option>
              <Option value="cross_section">Cross Section</Option>
              <Option value="logical">Logical</Option>
            </Select>
          </Space>
        </Col>
        <Col>
          <Button 
            icon={<SyncOutlined spin={syncMutation.isLoading} />} 
            onClick={() => syncMutation.mutate()}
            loading={syncMutation.isLoading}
          >
            同步算子库
          </Button>
        </Col>
      </Row>
      <Table
        columns={columns}
        dataSource={operators || []}
        rowKey="id"
        loading={isLoading}
        pagination={{ pageSize: 20 }}
      />
    </div>
  )
}

// Datasets Tab Component
const DatasetsTab = () => {
  const [search, setSearch] = useState('')
  const [filters, setFilters] = useState({ region: null, category: null })
  const [selectedDataset, setSelectedDataset] = useState(null)
  const [pagination, setPagination] = useState({ current: 1, pageSize: 12 })
  // Removed drawerVisible, replaced by conditional rendering
  const queryClient = useQueryClient()

  const { data: datasetsData, isLoading } = useQuery({
    queryKey: ['datasets', search, filters, pagination],
    queryFn: () => api.getDatasets({ 
        search, 
        ...filters,
        limit: pagination.pageSize,
        offset: (pagination.current - 1) * pagination.pageSize
    }),
    keepPreviousData: true,
  })

  // Fetch categories
  const { data: categories } = useQuery({
    queryKey: ['datasetCategories'],
    queryFn: api.getDatasetCategories,
  })

  const REGION_UNIVERSE_MAP = {
    'USA': 'TOP3000',
    'CHN': 'TOP3000',
    'EUR': 'TOP2500',
    'ASI': 'MINVOL1M',
    'GLB': 'TOPDIV3000',
    'KOR': 'TOP600',
    'IND': 'TOP500',
    'AMR': 'TOP600'
  }

  // Polling logic
  const [activeTask, setActiveTask] = useState(null) // { id: str, region: str }
  
  // Poll active task
  useQuery({
    queryKey: ['taskStatus', activeTask?.id],
    queryFn: () => api.getAsyncStatus(activeTask.id),
    enabled: !!activeTask?.id,
    refetchInterval: 2000,
    onSuccess: (data) => {
      if (data.status === 'SUCCESS') {
        message.success(`${activeTask.region} 同步完成！`)
        setActiveTask(null)
        queryClient.invalidateQueries(['datasets'])
      } else if (data.status === 'FAILURE' || data.status === 'REVOKED') {
        message.error(`同步失败: ${data.error || '未知错误'}`)
        setActiveTask(null)
      }
    }
  })

  const syncMutation = useMutation({
    mutationFn: ({ region }) => {
      const universe = REGION_UNIVERSE_MAP[region] || 'TOP3000'
      return api.syncDatasets(region, universe)
    },
    onSuccess: (data, variables) => {
      message.info(`后台同步已启动: ${data.message}`)
      if (data.task_id) {
        setActiveTask({ id: data.task_id, region: variables.region })
      }
    },
    onError: () => message.error('启动同步失败'),
  })

  const handleSync = (region) => {
    if (activeTask) {
        message.warning('已有同步任务正在进行中')
        return
    }
    syncMutation.mutate({ region })
  }

  const columns = [
    {
      title: '数据集 (Dataset)',
      dataIndex: 'dataset_id',
      key: 'dataset_id',
      width: 280,
      render: (text) => <Text strong style={{ color: '#1890ff', cursor: 'pointer' }}>{text}</Text>,
      onCell: (record) => ({
        onClick: () => setSelectedDataset(record),
      }),
    },
    {
      title: '字段数',
      dataIndex: 'field_count',
      key: 'field_count',
      width: 80,
      align: 'right',
    },
    {
      title: 'Pyramid',
      dataIndex: 'pyramid_multiplier',
      key: 'pyramid_multiplier',
      width: 100,
      align: 'right',
      render: val => val?.toFixed(1)
    },
    {
      title: '覆盖率',
      dataIndex: 'coverage',
      key: 'coverage',
      width: 100,
      align: 'right',
      render: val => val ? `${(val*100).toFixed(0)}%` : '-'
    },
    {
      title: '日期覆盖',
      dataIndex: 'date_coverage',
      key: 'date_coverage',
      width: 100,
      align: 'right',
      render: val => val ? `${(val*100).toFixed(0)}%` : '-'
    },
    {
      title: 'Value Score',
      dataIndex: 'value_score',
      key: 'value_score',
      width: 100,
      align: 'right',
    },
    {
      title: 'Alphas',
      dataIndex: 'alpha_count',
      key: 'alpha_count',
      width: 80,
      align: 'right',
    },
    {
      title: '资源',
      dataIndex: 'resources',
      key: 'resources',
      width: 80,
      align: 'center',
      render: (res) => (res && res.length > 0) ? <Tag color="purple">Paper</Tag> : null
    },
    {
      title: '地区',
      dataIndex: 'region',
      key: 'region',
      width: 80,
      render: (region) => <Tag color="blue">{region}</Tag>,
    },
    {
      title: 'Universe',
      dataIndex: 'universe',
      key: 'universe',
      width: 100,
    },
    {
      title: '类别',
      dataIndex: 'category',
      key: 'category',
      width: 120,
    },
    {
      title: '描述',
      dataIndex: 'description',
      key: 'description',
      ellipsis: true,
    },
    {
      title: '字段',
      dataIndex: 'field_count',
      key: 'field_count',
      width: 80,
      align: 'right',
    },
    {
      title: '上次同步',
      dataIndex: 'last_synced_at',
      key: 'last_synced_at',
      width: 150,
      render: (date) => (
        <Tooltip title={date ? new Date(date).toLocaleString() : 'Never'}>
          {date ? new Date(date).toLocaleDateString() : '-'}
        </Tooltip>
      ),
    },
    {
      title: '操作',
      key: 'action',
      width: 120,
      render: (_, record) => (
        <Button 
          type="link" 
          onClick={() => setSelectedDataset(record)}
        >
          查看详情
        </Button>
      )
    }
  ]

  if (selectedDataset) {
      return <DatasetDetailView dataset={selectedDataset} onBack={() => setSelectedDataset(null)} />
  }

  return (
    <>
      <Row justify="space-between" style={{ marginBottom: 16 }}>
        <Col span={16}>
          <Space>
            <Input 
              placeholder="搜索数据集..." 
              prefix={<SearchOutlined />} 
              onChange={e => setSearch(e.target.value)}
              style={{ width: 200 }}
              allowClear
            />
            <Select 
              placeholder="地区" 
              style={{ width: 120 }}
              allowClear
              onChange={val => setFilters(prev => ({ ...prev, region: val }))}
            >
              <Option value="USA">USA</Option>
              <Option value="CHN">China</Option>
              <Option value="ASI">Asia</Option>
              <Option value="GLB">Global</Option>
              <Option value="EUR">Europe</Option>
              <Option value="KOR">Korea</Option>
              <Option value="IND">India</Option>
              <Option value="AMR">America</Option>
            </Select>
            <Select 
              placeholder="类别" 
              style={{ width: 150 }}
              allowClear
              onChange={val => setFilters(prev => ({ ...prev, category: val }))}
            >
              {(categories || []).map(c => (
                <Option key={c} value={c}>{c}</Option>
              ))}
            </Select>
          </Space>
        </Col>
        <Col>
          <Space>
            <Button 
              icon={<SyncOutlined spin={activeTask?.region === 'USA'} />} 
              onClick={() => handleSync('USA')}
              loading={activeTask?.region === 'USA'}
            >
              USA
            </Button>
            <Button 
              icon={<SyncOutlined spin={activeTask?.region === 'CHN'} />} 
              onClick={() => handleSync('CHN')}
              loading={activeTask?.region === 'CHN'}
            >
              CHN
            </Button>
            <Button 
              icon={<SyncOutlined spin={activeTask?.region === 'ASI'} />} 
              onClick={() => handleSync('ASI')}
              loading={activeTask?.region === 'ASI'}
            >
              ASI
            </Button>
            <Button 
              icon={<SyncOutlined spin={activeTask?.region === 'GLB'} />} 
              onClick={() => handleSync('GLB')}
              loading={activeTask?.region === 'GLB'}
            >
              GLB
            </Button>
            <Button 
              icon={<SyncOutlined spin={activeTask?.region === 'EUR'} />} 
              onClick={() => handleSync('EUR')}
              loading={activeTask?.region === 'EUR'}
            >
              EUR
            </Button>
            <Button 
              icon={<SyncOutlined spin={activeTask?.region === 'KOR'} />} 
              onClick={() => handleSync('KOR')}
              loading={activeTask?.region === 'KOR'}
            >
              KOR
            </Button>
            <Button 
              icon={<SyncOutlined spin={activeTask?.region === 'IND'} />} 
              onClick={() => handleSync('IND')}
              loading={activeTask?.region === 'IND'}
            >
              IND
            </Button>
            <Button 
              icon={<SyncOutlined spin={activeTask?.region === 'AMR'} />} 
              onClick={() => handleSync('AMR')}
              loading={activeTask?.region === 'AMR'}
            >
              AMR
            </Button>
          </Space>
        </Col>
      </Row>

      <Table
        columns={columns}
        dataSource={datasetsData?.results || []}
        rowKey="dataset_id"
        loading={isLoading}
        pagination={{
            ...pagination,
            total: datasetsData?.total || 0,
            showTotal: (total) => `共 ${total} 条`,
            showSizeChanger: true,
            onChange: (page, pageSize) => setPagination({ current: page, pageSize })
        }}
      />
    </>
  )
}

// Main Page Component
export default function DataManagement() {
  const items = [
    {
      key: 'datasets',
      label: (
        <span>
          <DatabaseOutlined />
          数据集 (Datasets)
        </span>
      ),
      children: <DatasetsTab />,
    },
    {
      key: 'operators',
      label: (
        <span>
          <FunctionOutlined />
          算子库 (Operators)
        </span>
      ),
      children: <OperatorsTab />,
    },
  ]

  return (
    <div>
      <Row justify="space-between" align="middle" style={{ marginBottom: 24 }}>
        <Col>
          <Title level={3} style={{ margin: 0 }}>
            <DatabaseOutlined style={{ marginRight: 12, color: '#00d4ff' }} />
            数据与算子管理
          </Title>
        </Col>
      </Row>

      <Card className="glass-card" styles={{ body: { padding: '0 24px 24px 24px' } }}>
        <Tabs defaultActiveKey="datasets" items={items} size="large" />
      </Card>
    </div>
  )
}
