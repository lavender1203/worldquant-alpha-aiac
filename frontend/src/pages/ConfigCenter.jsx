import { useEffect } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { 
  Row, 
  Col, 
  Card, 
  Typography, 
  Tabs,
  Slider,
  Switch,
  Table,
  Tag,
  Button,
  Space,
  InputNumber,
  Form,
  Input,
  message,
  Alert,
  Spin,
  Tooltip,
  Divider,
} from 'antd'
import {
  SettingOutlined,
  SaveOutlined,
  KeyOutlined,
  CloudOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  SyncOutlined,
  EyeInvisibleOutlined,
  EyeTwoTone,
  ApiOutlined,
} from '@ant-design/icons'
import api from '../services/api'

const { Title, Text, Paragraph } = Typography

export default function ConfigCenter() {
  const queryClient = useQueryClient()
  const [brainForm] = Form.useForm()
  const [llmForm] = Form.useForm()
  const [thresholdForm] = Form.useForm()
  const sharpeMin = Form.useWatch('sharpe_min', thresholdForm)
  const turnoverMax = Form.useWatch('turnover_max', thresholdForm)
  const fitnessMin = Form.useWatch('fitness_min', thresholdForm)
  const maxCorrelation = Form.useWatch('max_correlation', thresholdForm)

  // Fetch knowledge entries
  const { data: successPatterns, isLoading: patternsLoading } = useQuery({
    queryKey: ['knowledge', 'success-patterns'],
    queryFn: () => api.getSuccessPatterns(30),
  })

  const { data: failurePitfalls, isLoading: pitfallsLoading } = useQuery({
    queryKey: ['knowledge', 'failure-pitfalls'],
    queryFn: () => api.getFailurePitfalls(30),
  })

  // Fetch credentials status
  const { data: credentialsData, isLoading: credentialsLoading, refetch: refetchCredentials } = useQuery({
    queryKey: ['credentials'],
    queryFn: api.getCredentialsStatus,
  })

  const { data: thresholdsData, isLoading: thresholdsLoading } = useQuery({
    queryKey: ['config', 'thresholds'],
    queryFn: api.getThresholds,
  })

  const { data: diversityData, isLoading: diversityLoading } = useQuery({
    queryKey: ['config', 'diversity'],
    queryFn: api.getDiversity,
  })

  const { data: operatorPrefs, isLoading: operatorPrefsLoading } = useQuery({
    queryKey: ['config', 'operators'],
    queryFn: api.getOperatorPrefs,
  })

  useEffect(() => {
    if (!thresholdsData && !diversityData) return

    thresholdForm.setFieldsValue({
      sharpe_min: thresholdsData?.sharpe_min ?? 1.58,
      turnover_max: thresholdsData?.turnover_max ?? 0.3,
      fitness_min: thresholdsData?.fitness_min ?? 1.0,
      max_correlation: diversityData?.max_correlation ?? 0.7,
    })
  }, [thresholdForm, thresholdsData, diversityData])

  // Mutations for credentials
  const saveBrainCredentialsMutation = useMutation({
    mutationFn: ({ email, password }) => api.setBrainCredentials(email, password),
    onSuccess: () => {
      message.success('Brain 平台凭证保存成功')
      refetchCredentials()
      brainForm.resetFields()
    },
    onError: (error) => {
      message.error(`保存失败: ${error.response?.data?.detail || error.message}`)
    },
  })

  const saveLLMCredentialsMutation = useMutation({
    mutationFn: ({ apiKey, baseUrl, model }) => api.setLLMCredentials(apiKey, baseUrl, model),
    onSuccess: () => {
      message.success('LLM API 凭证保存成功')
      refetchCredentials()
      llmForm.resetFields()
    },
    onError: (error) => {
      message.error(`保存失败: ${error.response?.data?.detail || error.message}`)
    },
  })

  const testBrainCredentialsMutation = useMutation({
    mutationFn: api.testBrainCredentials,
    onSuccess: () => {
      message.success('Brain 平台连接测试成功！')
    },
    onError: (error) => {
      message.error(`连接测试失败: ${error.response?.data?.detail || error.message}`)
    },
  })

  const saveThresholdsMutation = useMutation({
    mutationFn: async (values) => {
      await api.updateThresholds({
        sharpe_min: values.sharpe_min,
        turnover_max: values.turnover_max,
        fitness_min: values.fitness_min,
        returns_min: thresholdsData?.returns_min ?? 0,
        max_dd_max: thresholdsData?.max_dd_max ?? 0.3,
      })
      await api.updateDiversity({ max_correlation: values.max_correlation })
    },
    onSuccess: () => {
      message.success('质量阈值保存成功')
      queryClient.invalidateQueries({ queryKey: ['config', 'thresholds'] })
      queryClient.invalidateQueries({ queryKey: ['config', 'diversity'] })
    },
    onError: (error) => {
      message.error(`保存失败: ${error.response?.data?.detail || error.message}`)
    },
  })

  const updateOperatorMutation = useMutation({
    mutationFn: ({ operatorName, status }) => api.updateOperatorPref(operatorName, status),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['config', 'operators'] })
      queryClient.invalidateQueries({ queryKey: ['operators'] })
    },
    onError: (error) => {
      message.error(`更新失败: ${error.response?.data?.detail || error.message}`)
    },
  })

  const knowledgeColumns = [
    {
      title: '模式',
      dataIndex: 'pattern',
      key: 'pattern',
      width: 200,
    },
    {
      title: '描述',
      dataIndex: 'description',
      key: 'description',
      ellipsis: true,
    },
    {
      title: '使用次数',
      dataIndex: 'usage_count',
      key: 'usage_count',
      width: 80,
    },
    {
      title: '状态',
      dataIndex: 'is_active',
      key: 'is_active',
      width: 80,
      render: (active) => (
        <Tag color={active ? 'success' : 'default'}>
          {active ? 'Active' : 'Inactive'}
        </Tag>
      ),
    },
    {
      title: '来源',
      dataIndex: 'created_by',
      key: 'created_by',
      width: 80,
      render: (source) => (
        <Tag color={source === 'USER' ? 'blue' : 'default'}>{source}</Tag>
      ),
    },
  ]

  // Credentials tab content
  const CredentialsTab = () => {
    const credentials = credentialsData?.credentials || {}

    const renderCredentialStatus = (key, label) => {
      const cred = credentials[key] || {}
      const isSet = cred.is_set
      const source = cred.source
      
      return (
        <div style={{ 
          display: 'flex', 
          justifyContent: 'space-between', 
          alignItems: 'center',
          padding: '8px 0',
          borderBottom: '1px solid rgba(255,255,255,0.1)'
        }}>
          <Text>{label}</Text>
          <Space>
            {isSet ? (
              <>
                <Text type="secondary" style={{ fontFamily: 'monospace' }}>
                  {cred.masked}
                </Text>
                {source === 'env' && (
                  <Tooltip title="从环境变量读取">
                    <Tag color="blue">ENV</Tag>
                  </Tooltip>
                )}
                <CheckCircleOutlined style={{ color: '#52c41a' }} />
              </>
            ) : (
              <>
                <Text type="secondary">(未配置)</Text>
                <CloseCircleOutlined style={{ color: '#ff4d4f' }} />
              </>
            )}
          </Space>
        </div>
      )
    }

    return (
      <Row gutter={24}>
        {/* Brain Platform Credentials */}
        <Col xs={24} lg={12}>
          <Card 
            className="glass-card" 
            title={
              <Space>
                <CloudOutlined style={{ color: '#00d4ff' }} />
                <span>WorldQuant Brain 平台</span>
              </Space>
            }
          >
            <Alert
              message="Brain 平台凭证"
              description="用于连接 WorldQuant Brain 平台进行 Alpha 模拟和数据同步。"
              type="info"
              showIcon
              style={{ marginBottom: 16 }}
            />

            {credentialsLoading ? (
              <Spin />
            ) : (
              <div style={{ marginBottom: 24 }}>
                <Title level={5}>当前状态</Title>
                {renderCredentialStatus('brain_email', '邮箱')}
                {renderCredentialStatus('brain_password', '密码')}
              </div>
            )}

            <Divider />

            <Title level={5}>更新凭证</Title>
            <Form
              form={brainForm}
              layout="vertical"
              onFinish={(values) => {
                saveBrainCredentialsMutation.mutate(values)
              }}
            >
              <Form.Item
                name="email"
                label="Brain 平台邮箱"
                rules={[
                  { required: true, message: '请输入邮箱' },
                  { type: 'email', message: '请输入有效的邮箱地址' }
                ]}
              >
                <Input 
                  prefix={<KeyOutlined />} 
                  placeholder="your-email@example.com" 
                />
              </Form.Item>

              <Form.Item
                name="password"
                label="Brain 平台密码"
                rules={[{ required: true, message: '请输入密码' }]}
              >
                <Input.Password 
                  prefix={<KeyOutlined />}
                  placeholder="输入密码"
                  iconRender={(visible) => (visible ? <EyeTwoTone /> : <EyeInvisibleOutlined />)}
                />
              </Form.Item>

              <Form.Item>
                <Space>
                  <Button 
                    type="primary" 
                    htmlType="submit"
                    icon={<SaveOutlined />}
                    loading={saveBrainCredentialsMutation.isPending}
                  >
                    保存凭证
                  </Button>
                  <Button 
                    icon={<SyncOutlined />}
                    onClick={() => testBrainCredentialsMutation.mutate()}
                    loading={testBrainCredentialsMutation.isPending}
                  >
                    测试连接
                  </Button>
                </Space>
              </Form.Item>
            </Form>
          </Card>
        </Col>

        {/* LLM API Credentials */}
        <Col xs={24} lg={12}>
          <Card 
            className="glass-card"
            title={
              <Space>
                <ApiOutlined style={{ color: '#00d4ff' }} />
                <span>LLM API 配置</span>
              </Space>
            }
          >
            <Alert
              message="大语言模型 API"
              description="支持 OpenAI、DeepSeek、智谱等兼容 OpenAI 协议的 API 服务。"
              type="info"
              showIcon
              style={{ marginBottom: 16 }}
            />

            {credentialsLoading ? (
              <Spin />
            ) : (
              <div style={{ marginBottom: 24 }}>
                <Title level={5}>当前状态</Title>
                {renderCredentialStatus('openai_api_key', 'API Key')}
                {renderCredentialStatus('openai_base_url', 'Base URL')}
                {renderCredentialStatus('openai_model', '模型')}
              </div>
            )}

            <Divider />

            <Title level={5}>更新配置</Title>
            <Form
              form={llmForm}
              layout="vertical"
              initialValues={{
                baseUrl: 'https://api.deepseek.com/v1',
                model: 'deepseek-chat'
              }}
              onFinish={(values) => {
                saveLLMCredentialsMutation.mutate({
                  apiKey: values.apiKey,
                  baseUrl: values.baseUrl,
                  model: values.model
                })
              }}
            >
              <Form.Item
                name="apiKey"
                label="API Key"
                rules={[{ required: true, message: '请输入 API Key' }]}
              >
                <Input.Password 
                  prefix={<KeyOutlined />}
                  placeholder="sk-xxxxxxxxxxxxxxxx"
                  iconRender={(visible) => (visible ? <EyeTwoTone /> : <EyeInvisibleOutlined />)}
                />
              </Form.Item>

              <Form.Item
                name="baseUrl"
                label="Base URL"
                rules={[{ required: true, message: '请输入 Base URL' }]}
              >
                <Input 
                  placeholder="https://api.deepseek.com/v1" 
                />
              </Form.Item>

              <Form.Item
                name="model"
                label="模型名称"
                rules={[{ required: true, message: '请输入模型名称' }]}
              >
                <Input 
                  placeholder="deepseek-chat" 
                />
              </Form.Item>

              <Form.Item>
                <Button 
                  type="primary" 
                  htmlType="submit"
                  icon={<SaveOutlined />}
                  loading={saveLLMCredentialsMutation.isPending}
                >
                  保存配置
                </Button>
              </Form.Item>
            </Form>

            <Paragraph type="secondary" style={{ marginTop: 16 }}>
              <Text strong>常用 API 地址:</Text>
              <ul style={{ marginTop: 8 }}>
                <li>DeepSeek: https://api.deepseek.com/v1</li>
                <li>OpenAI: https://api.openai.com/v1</li>
                <li>智谱: https://open.bigmodel.cn/api/paas/v4</li>
                <li>Moonshot: https://api.moonshot.cn/v1</li>
              </ul>
            </Paragraph>
          </Card>
        </Col>
      </Row>
    )
  }

  const tabs = [
    {
      key: 'credentials',
      label: (
        <Space>
          <KeyOutlined />
          凭证管理
        </Space>
      ),
      children: <CredentialsTab />,
    },
    {
      key: 'thresholds',
      label: '质量阈值',
      children: (
        <Card className="glass-card">
          <Spin spinning={thresholdsLoading || diversityLoading}>
          <Form
            form={thresholdForm}
            layout="vertical"
            style={{ maxWidth: 500 }}
            initialValues={{
              sharpe_min: 1.58,
              turnover_max: 0.3,
              fitness_min: 1.0,
              max_correlation: 0.7,
            }}
          >
            <Form.Item label="最低夏普比率 (Sharpe Ratio)">
              <Row gutter={16}>
                <Col span={16}>
                  <Slider 
                    min={0} 
                    max={5} 
                    step={0.1} 
                    value={sharpeMin}
                    onChange={(value) => thresholdForm.setFieldValue('sharpe_min', value)}
                    marks={{ 0: '0', 1: '1', 1.58: '1.58', 2: '2', 3: '3', 5: '5' }}
                  />
                </Col>
                <Col span={8}>
                  <InputNumber min={0} max={5} step={0.1} value={sharpeMin} onChange={(value) => thresholdForm.setFieldValue('sharpe_min', value)} style={{ width: '100%' }} />
                </Col>
              </Row>
            </Form.Item>

            <Form.Item label="最高换手率 (Turnover)">
              <Row gutter={16}>
                <Col span={16}>
                  <Slider 
                    min={0} 
                    max={2} 
                    step={0.1} 
                    value={turnoverMax}
                    onChange={(value) => thresholdForm.setFieldValue('turnover_max', value)}
                    marks={{ 0: '0', 0.3: '0.3', 0.7: '0.7', 1: '1', 1.5: '1.5', 2: '2' }}
                  />
                </Col>
                <Col span={8}>
                  <InputNumber min={0} max={2} step={0.1} value={turnoverMax} onChange={(value) => thresholdForm.setFieldValue('turnover_max', value)} style={{ width: '100%' }} />
                </Col>
              </Row>
            </Form.Item>

            <Form.Item label="最低适应度 (Fitness)">
              <Row gutter={16}>
                <Col span={16}>
                  <Slider 
                    min={0} 
                    max={3}
                    step={0.05} 
                    value={fitnessMin}
                    onChange={(value) => thresholdForm.setFieldValue('fitness_min', value)}
                    marks={{ 0: '0', 0.6: '0.6', 1: '1', 2: '2', 3: '3' }}
                  />
                </Col>
                <Col span={8}>
                  <InputNumber min={0} max={3} step={0.05} value={fitnessMin} onChange={(value) => thresholdForm.setFieldValue('fitness_min', value)} style={{ width: '100%' }} />
                </Col>
              </Row>
            </Form.Item>

            <Form.Item label="最大相关性 (多样性)">
              <Row gutter={16}>
                <Col span={16}>
                  <Slider 
                    min={0} 
                    max={1} 
                    step={0.05} 
                    value={maxCorrelation}
                    onChange={(value) => thresholdForm.setFieldValue('max_correlation', value)}
                    marks={{ 0: '0', 0.5: '0.5', 0.7: '0.7', 1: '1' }}
                  />
                </Col>
                <Col span={8}>
                  <InputNumber min={0} max={1} step={0.05} value={maxCorrelation} onChange={(value) => thresholdForm.setFieldValue('max_correlation', value)} style={{ width: '100%' }} />
                </Col>
              </Row>
            </Form.Item>

            <Form.Item>
              <Button
                type="primary"
                icon={<SaveOutlined />}
                loading={saveThresholdsMutation.isPending}
                onClick={() => saveThresholdsMutation.mutate(thresholdForm.getFieldsValue(true))}
              >
                保存设置
              </Button>
            </Form.Item>
          </Form>
          </Spin>
        </Card>
      ),
    },
    {
      key: 'operators',
      label: '算子偏好',
      children: (
        <Card className="glass-card">
          <Table
            dataSource={operatorPrefs || []}
            loading={operatorPrefsLoading}
            columns={[
              { title: '算子', dataIndex: 'operator_name', key: 'operator_name' },
              { title: '使用次数', dataIndex: 'usage_count', key: 'usage_count' },
              { 
                title: '成功率', 
                key: 'success_rate',
                render: (_, record) => {
                  const failureRate = record.failure_rate > 1
                    ? record.failure_rate / 100
                    : (record.failure_rate || 0)
                  const rate = record.usage_count > 0
                    ? Math.round((record.success_count / record.usage_count) * 100)
                    : Math.round((1 - failureRate) * 100)
                  return (
                    <Text style={{ color: rate > 50 ? '#00ff88' : '#ff4757' }}>
                      {rate}%
                    </Text>
                  )
                },
              },
              { 
                title: '状态', 
                dataIndex: 'status', 
                key: 'status',
                render: (status) => (
                  <Tag color={status === 'ACTIVE' ? 'success' : 'error'}>{status}</Tag>
                ),
              },
              {
                title: '操作',
                key: 'action',
                render: (_, record) => (
                  <Switch 
                    checked={record.status === 'ACTIVE'} 
                    checkedChildren="启用"
                    unCheckedChildren="禁用"
                    loading={updateOperatorMutation.isPending}
                    onChange={(checked) => updateOperatorMutation.mutate({
                      operatorName: record.operator_name,
                      status: checked ? 'ACTIVE' : 'BANNED',
                    })}
                  />
                ),
              },
            ]}
            rowKey="operator_name"
            pagination={{ pageSize: 20 }}
          />
        </Card>
      ),
    },
    {
      key: 'success-patterns',
      label: '成功模式',
      children: (
        <Card className="glass-card">
          <Table
            columns={knowledgeColumns}
            dataSource={successPatterns || []}
            rowKey="id"
            loading={patternsLoading}
            pagination={{ pageSize: 10 }}
          />
        </Card>
      ),
    },
    {
      key: 'failure-pitfalls',
      label: '失败教训',
      children: (
        <Card className="glass-card">
          <Table
            columns={knowledgeColumns}
            dataSource={failurePitfalls || []}
            rowKey="id"
            loading={pitfallsLoading}
            pagination={{ pageSize: 10 }}
          />
        </Card>
      ),
    },
  ]

  return (
    <div>
      <Title level={3} style={{ marginBottom: 24 }}>
        <SettingOutlined style={{ marginRight: 12, color: '#00d4ff' }} />
        配置中心
      </Title>

      <Tabs items={tabs} size="large" defaultActiveKey="credentials" />
    </div>
  )
}
