import { useMemo, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import {
  Alert,
  Button,
  Card,
  Col,
  Divider,
  Form,
  Input,
  Modal,
  Popconfirm,
  Row,
  Space,
  Switch,
  Table,
  Tag,
  Tooltip,
  Typography,
  message,
} from 'antd'
import {
  ApiOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  DeleteOutlined,
  EditOutlined,
  PlusOutlined,
  ReloadOutlined,
  ToolOutlined,
} from '@ant-design/icons'
import api from '../services/api'

const { Title, Text, Paragraph } = Typography
const { TextArea } = Input

const defaultServer = {
  name: 'local-mcp',
  url: 'http://172.0.0.1:8876/mcp',
  transport: 'streamable_http',
  description: '',
  headersText: '{}',
  is_enabled: true,
}

function parseHeaders(value) {
  if (!value || !value.trim()) return {}
  try {
    const parsed = JSON.parse(value)
    if (!parsed || Array.isArray(parsed) || typeof parsed !== 'object') {
      throw new Error('headers 必须是 JSON object')
    }
    return parsed
  } catch (error) {
    throw new Error(`Headers JSON 无效: ${error.message}`)
  }
}

function statusTag(status) {
  if (status === 'CONNECTED') return <Tag color="success">CONNECTED</Tag>
  if (status === 'FAILED') return <Tag color="error">FAILED</Tag>
  return <Tag>UNKNOWN</Tag>
}

export default function MCPManagement() {
  const queryClient = useQueryClient()
  const [form] = Form.useForm()
  const [editingServer, setEditingServer] = useState(null)
  const [modalOpen, setModalOpen] = useState(false)
  const [testResult, setTestResult] = useState(null)

  const { data: servers = [], isLoading } = useQuery({
    queryKey: ['mcp-servers'],
    queryFn: api.getMCPServers,
  })

  const invalidate = () => queryClient.invalidateQueries({ queryKey: ['mcp-servers'] })

  const createMutation = useMutation({
    mutationFn: api.createMCPServer,
    onSuccess: () => {
      message.success('MCP 已新增')
      setModalOpen(false)
      setEditingServer(null)
      form.resetFields()
      invalidate()
    },
    onError: (error) => message.error(error.response?.data?.detail || error.message),
  })

  const updateMutation = useMutation({
    mutationFn: ({ id, updates }) => api.updateMCPServer(id, updates),
    onSuccess: () => {
      message.success('MCP 配置已保存')
      setModalOpen(false)
      setEditingServer(null)
      form.resetFields()
      invalidate()
    },
    onError: (error) => message.error(error.response?.data?.detail || error.message),
  })

  const deleteMutation = useMutation({
    mutationFn: api.deleteMCPServer,
    onSuccess: () => {
      message.success('MCP 已删除')
      invalidate()
    },
    onError: (error) => message.error(error.response?.data?.detail || error.message),
  })

  const toggleServerMutation = useMutation({
    mutationFn: ({ id, enabled }) => api.updateMCPServer(id, { is_enabled: enabled }),
    onSuccess: invalidate,
    onError: (error) => message.error(error.response?.data?.detail || error.message),
  })

  const testMutation = useMutation({
    mutationFn: api.testMCPServer,
    onSuccess: (data) => {
      setTestResult(data)
      if (data.ok) {
        message.success(`连接成功，发现 ${data.tools?.length || 0} 个函数`)
      } else {
        message.error(data.message || '连接失败')
      }
      invalidate()
    },
    onError: (error) => message.error(error.response?.data?.detail || error.message),
  })

  const refreshMutation = useMutation({
    mutationFn: api.refreshMCPTools,
    onSuccess: (data) => {
      setTestResult(data)
      message.success(`工具已刷新：${data.tools?.length || 0} 个`)
      invalidate()
    },
    onError: (error) => message.error(error.response?.data?.detail || error.message),
  })

  const toggleToolMutation = useMutation({
    mutationFn: ({ id, enabled }) => api.updateMCPTool(id, enabled),
    onSuccess: invalidate,
    onError: (error) => message.error(error.response?.data?.detail || error.message),
  })

  const openCreate = () => {
    setEditingServer(null)
    setTestResult(null)
    form.setFieldsValue(defaultServer)
    setModalOpen(true)
  }

  const openEdit = (server) => {
    setEditingServer(server)
    setTestResult(null)
    form.setFieldsValue({
      name: server.name,
      url: server.url,
      transport: server.transport || 'streamable_http',
      description: server.description || '',
      headersText: JSON.stringify(server.headers || {}, null, 2),
      is_enabled: server.is_enabled,
    })
    setModalOpen(true)
  }

  const submit = async () => {
    const values = await form.validateFields()
    let headers = {}
    try {
      headers = parseHeaders(values.headersText)
    } catch (error) {
      message.error(error.message)
      return
    }

    const payload = {
      name: values.name,
      url: values.url,
      transport: values.transport || 'streamable_http',
      description: values.description || null,
      headers,
      is_enabled: values.is_enabled,
    }

    if (editingServer) {
      updateMutation.mutate({ id: editingServer.id, updates: payload })
    } else {
      createMutation.mutate(payload)
    }
  }

  const totals = useMemo(() => {
    const tools = servers.flatMap((server) => server.tools || [])
    return {
      servers: servers.length,
      enabledServers: servers.filter((server) => server.is_enabled).length,
      tools: tools.length,
      enabledTools: tools.filter((tool) => tool.is_enabled).length,
    }
  }, [servers])

  const toolColumns = [
    {
      title: '函数',
      dataIndex: 'name',
      key: 'name',
      width: 260,
      render: (name, record) => (
        <Space direction="vertical" size={2}>
          <Text strong>{name}</Text>
          {record.description && <Text type="secondary">{record.description}</Text>}
        </Space>
      ),
    },
    {
      title: 'Schema',
      dataIndex: 'input_schema',
      key: 'input_schema',
      ellipsis: true,
      render: (schema) => (
        <Tooltip title={<pre style={{ margin: 0 }}>{JSON.stringify(schema || {}, null, 2)}</pre>}>
          <Text code>{Object.keys(schema?.properties || {}).join(', ') || 'no args'}</Text>
        </Tooltip>
      ),
    },
    {
      title: '启用',
      dataIndex: 'is_enabled',
      key: 'is_enabled',
      width: 90,
      render: (enabled, record) => (
        <Switch
          size="small"
          checked={enabled}
          onChange={(checked) => toggleToolMutation.mutate({ id: record.id, enabled: checked })}
        />
      ),
    },
  ]

  const serverColumns = [
    {
      title: 'MCP Server',
      dataIndex: 'name',
      key: 'name',
      width: 260,
      render: (_, record) => (
        <Space direction="vertical" size={2}>
          <Space>
            <ApiOutlined />
            <Text strong>{record.name}</Text>
            {record.is_enabled ? <Tag color="blue">ON</Tag> : <Tag>OFF</Tag>}
          </Space>
          <Text type="secondary" copyable style={{ maxWidth: 520 }}>{record.url}</Text>
          {record.description && <Text type="secondary">{record.description}</Text>}
        </Space>
      ),
    },
    {
      title: '状态',
      dataIndex: 'last_status',
      key: 'last_status',
      width: 150,
      render: (_, record) => (
        <Space direction="vertical" size={2}>
          {statusTag(record.last_status)}
          {record.last_error && (
            <Tooltip title={record.last_error}>
              <Text type="danger">查看错误</Text>
            </Tooltip>
          )}
        </Space>
      ),
    },
    {
      title: '函数',
      key: 'tools',
      width: 120,
      render: (_, record) => {
        const tools = record.tools || []
        const enabled = tools.filter((tool) => tool.is_enabled).length
        return <Tag color={enabled ? 'green' : 'default'}>{enabled}/{tools.length}</Tag>
      },
    },
    {
      title: '启用',
      dataIndex: 'is_enabled',
      key: 'is_enabled',
      width: 90,
      render: (enabled, record) => (
        <Switch
          checked={enabled}
          onChange={(checked) => toggleServerMutation.mutate({ id: record.id, enabled: checked })}
        />
      ),
    },
    {
      title: '操作',
      key: 'actions',
      width: 280,
      render: (_, record) => (
        <Space wrap>
          <Button
            icon={<CheckCircleOutlined />}
            onClick={() => testMutation.mutate(record.id)}
            loading={testMutation.isPending}
          >
            测试
          </Button>
          <Button
            icon={<ReloadOutlined />}
            onClick={() => refreshMutation.mutate(record.id)}
            loading={refreshMutation.isPending}
          >
            刷新函数
          </Button>
          <Button icon={<EditOutlined />} onClick={() => openEdit(record)} />
          <Popconfirm title="删除该 MCP？" onConfirm={() => deleteMutation.mutate(record.id)}>
            <Button danger icon={<DeleteOutlined />} />
          </Popconfirm>
        </Space>
      ),
    },
  ]

  return (
    <div>
      <Row gutter={[16, 16]} align="middle" style={{ marginBottom: 16 }}>
        <Col flex="auto">
          <Title level={2} style={{ margin: 0 }}>MCP 管理</Title>
          <Paragraph type="secondary" style={{ marginBottom: 0 }}>
            配置外部 MCP endpoint，测试连接，并控制 server 与函数级别的启用状态。
          </Paragraph>
        </Col>
        <Col>
          <Button type="primary" icon={<PlusOutlined />} onClick={openCreate}>新增 MCP</Button>
        </Col>
      </Row>

      <Row gutter={[16, 16]} style={{ marginBottom: 16 }}>
        <Col xs={24} md={6}>
          <Card><Text type="secondary">Servers</Text><Title level={3}>{totals.enabledServers}/{totals.servers}</Title></Card>
        </Col>
        <Col xs={24} md={6}>
          <Card><Text type="secondary">Functions</Text><Title level={3}>{totals.enabledTools}/{totals.tools}</Title></Card>
        </Col>
        <Col xs={24} md={12}>
          <Alert
            type="info"
            showIcon
            message="Docker 内连接本机 MCP"
            description="如果 MCP 跑在宿主机 localhost，后端会自动尝试 host.docker.internal 和 172.17.0.1 作为 Docker 访问宿主机的候选地址。"
          />
        </Col>
      </Row>

      {testResult && (
        <Alert
          style={{ marginBottom: 16 }}
          type={testResult.ok ? 'success' : 'error'}
          showIcon
          icon={testResult.ok ? <CheckCircleOutlined /> : <CloseCircleOutlined />}
          message={testResult.message}
          description={
            <Space direction="vertical" size={2}>
              <Text>尝试地址：{(testResult.tried_urls || []).join(' , ')}</Text>
              <Text>发现函数：{(testResult.tools || []).map((tool) => tool.name).join(', ') || '无'}</Text>
            </Space>
          }
          closable
          onClose={() => setTestResult(null)}
        />
      )}

      <Card>
        <Table
          rowKey="id"
          loading={isLoading}
          columns={serverColumns}
          dataSource={servers}
          expandable={{
            expandedRowRender: (record) => (
              <Table
                rowKey="id"
                size="small"
                pagination={false}
                columns={toolColumns}
                dataSource={record.tools || []}
                locale={{ emptyText: '还没有发现函数，先点击“测试”或“刷新函数”。' }}
              />
            ),
          }}
        />
      </Card>

      <Modal
        title={editingServer ? '编辑 MCP' : '新增 MCP'}
        open={modalOpen}
        onCancel={() => setModalOpen(false)}
        onOk={submit}
        confirmLoading={createMutation.isPending || updateMutation.isPending}
        width={720}
      >
        <Form form={form} layout="vertical" initialValues={defaultServer}>
          <Row gutter={16}>
            <Col span={12}>
              <Form.Item name="name" label="名称" rules={[{ required: true, message: '请输入名称' }]}>
                <Input prefix={<ToolOutlined />} placeholder="local-mcp" />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item name="transport" label="Transport">
                <Input placeholder="streamable_http" />
              </Form.Item>
            </Col>
          </Row>
          <Form.Item name="url" label="Endpoint" rules={[{ required: true, message: '请输入 MCP endpoint' }]}>
            <Input placeholder="http://172.0.0.1:8876/mcp" />
          </Form.Item>
          <Form.Item name="description" label="描述">
            <Input placeholder="用途、来源或备注" />
          </Form.Item>
          <Form.Item name="headersText" label="Headers JSON">
            <TextArea rows={5} spellCheck={false} />
          </Form.Item>
          <Divider />
          <Form.Item name="is_enabled" label="启用 Server" valuePropName="checked">
            <Switch />
          </Form.Item>
        </Form>
      </Modal>
    </div>
  )
}
