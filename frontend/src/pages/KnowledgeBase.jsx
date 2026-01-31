import { useState, useCallback } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { Document, Page, pdfjs } from 'react-pdf'
import 'react-pdf/dist/Page/AnnotationLayer.css'
import 'react-pdf/dist/Page/TextLayer.css'

// 设置 PDF.js worker
pdfjs.GlobalWorkerOptions.workerSrc = `//unpkg.com/pdfjs-dist@${pdfjs.version}/build/pdf.worker.min.mjs`
import { 
  Row, 
  Col, 
  Card, 
  Typography, 
  Tabs,
  Table,
  Tag,
  Button,
  Space,
  Form,
  Input,
  Select,
  message,
  Alert,
  Spin,
  Modal,
  Tooltip,
  Divider,
  Statistic,
  Badge,
  Popconfirm,
  Empty,
} from 'antd'
import {
  BookOutlined,
  SaveOutlined,
  SyncOutlined,
  DeleteOutlined,
  EditOutlined,
  PlusOutlined,
  SearchOutlined,
  CloudDownloadOutlined,
  FileTextOutlined,
  ExperimentOutlined,
  GlobalOutlined,
  RobotOutlined,
  LinkOutlined,
  DownloadOutlined,
  EyeOutlined,
  FilePdfOutlined,
  ReadOutlined,
  ExpandOutlined,
  LeftOutlined,
  RightOutlined,
  ZoomInOutlined,
  ZoomOutOutlined,
  FullscreenOutlined,
} from '@ant-design/icons'
import api from '../services/api'

const { Title, Text, Paragraph } = Typography
const { TextArea } = Input
const { Option } = Select

export default function KnowledgeBase() {
  const queryClient = useQueryClient()
  const [addForm] = Form.useForm()
  const [editForm] = Form.useForm()
  const [paperForm] = Form.useForm()
  const [forumSearchQuery, setForumSearchQuery] = useState('')
  const [editModalVisible, setEditModalVisible] = useState(false)
  const [editingEntry, setEditingEntry] = useState(null)
  const [addModalVisible, setAddModalVisible] = useState(false)
  const [paperModalVisible, setPaperModalVisible] = useState(false)
  const [forumPosts, setForumPosts] = useState([])
  const [forumLoading, setForumLoading] = useState(false)
  const [pdfViewerVisible, setPdfViewerVisible] = useState(false)
  const [currentPdf, setCurrentPdf] = useState(null)
  const [articleViewerVisible, setArticleViewerVisible] = useState(false)
  const [currentArticle, setCurrentArticle] = useState(null)
  const [numPages, setNumPages] = useState(null)
  const [currentPage, setCurrentPage] = useState(1)
  const [pdfScale, setPdfScale] = useState(1.2)

  // Fetch knowledge stats
  const { data: statsData, isLoading: statsLoading } = useQuery({
    queryKey: ['knowledge-stats'],
    queryFn: api.getKnowledgeStats,
  })

  // Fetch external knowledge
  const { data: externalKnowledge, isLoading: externalLoading, refetch: refetchExternal } = useQuery({
    queryKey: ['knowledge-external'],
    queryFn: () => api.getExternalKnowledge(),
  })

  // Fetch system knowledge
  const { data: systemKnowledge, isLoading: systemLoading, refetch: refetchSystem } = useQuery({
    queryKey: ['knowledge-system'],
    queryFn: () => api.getSystemKnowledge(),
  })

  // Fetch papers
  const { data: papersData, isLoading: papersLoading, refetch: refetchPapers } = useQuery({
    queryKey: ['knowledge-papers'],
    queryFn: api.getPapers,
  })

  // Create knowledge mutation
  const createMutation = useMutation({
    mutationFn: (data) => api.createKnowledge(data),
    onSuccess: () => {
      message.success('知识条目创建成功')
      queryClient.invalidateQueries(['knowledge'])
      refetchExternal()
      refetchSystem()
      setAddModalVisible(false)
      addForm.resetFields()
    },
    onError: (error) => {
      message.error(`创建失败: ${error.response?.data?.detail || error.message}`)
    },
  })

  // Update knowledge mutation
  const updateMutation = useMutation({
    mutationFn: ({ id, data }) => api.updateKnowledge(id, data),
    onSuccess: () => {
      message.success('知识条目更新成功')
      queryClient.invalidateQueries(['knowledge'])
      refetchExternal()
      refetchSystem()
      setEditModalVisible(false)
      setEditingEntry(null)
    },
    onError: (error) => {
      message.error(`更新失败: ${error.response?.data?.detail || error.message}`)
    },
  })

  // Delete knowledge mutation
  const deleteMutation = useMutation({
    mutationFn: (id) => api.deleteKnowledge(id),
    onSuccess: () => {
      message.success('知识条目已删除')
      queryClient.invalidateQueries(['knowledge'])
      refetchExternal()
      refetchSystem()
    },
    onError: (error) => {
      message.error(`删除失败: ${error.response?.data?.detail || error.message}`)
    },
  })

  // Download paper mutation
  const downloadPaperMutation = useMutation({
    mutationFn: (data) => api.downloadPaper(data),
    onSuccess: (result) => {
      message.success(result.message)
      refetchPapers()
      setPaperModalVisible(false)
      paperForm.resetFields()
    },
    onError: (error) => {
      message.error(`下载失败: ${error.response?.data?.detail || error.message}`)
    },
  })

  // Forum sync mutation
  const forumSyncMutation = useMutation({
    mutationFn: (data) => api.syncForum(data),
    onSuccess: () => {
      message.success('论坛同步已启动')
      refetchExternal()
    },
    onError: (error) => {
      message.error(`同步失败: ${error.response?.data?.detail || error.message}`)
    },
  })

  // Import forum post mutation
  const importForumMutation = useMutation({
    mutationFn: (post) => api.importForumPost(post),
    onSuccess: (result) => {
      message.success(result.message)
      refetchExternal()
    },
    onError: (error) => {
      message.error(`导入失败: ${error.response?.data?.detail || error.message}`)
    },
  })

  // Search forum
  const handleForumSearch = async () => {
    if (!forumSearchQuery.trim()) return
    
    setForumLoading(true)
    try {
      const posts = await api.searchForum(forumSearchQuery)
      setForumPosts(posts)
    } catch (error) {
      message.error('搜索失败')
    }
    setForumLoading(false)
  }

  // Edit entry
  const handleEdit = (entry) => {
    setEditingEntry(entry)
    editForm.setFieldsValue({
      pattern: entry.pattern,
      description: entry.description,
      is_active: entry.is_active,
    })
    setEditModalVisible(true)
  }

  // View PDF
  const handleViewPdf = (paper) => {
    setCurrentPdf(paper)
    setCurrentPage(1)
    setNumPages(null)
    setPdfViewerVisible(true)
  }

  // PDF load success callback
  const onPdfLoadSuccess = useCallback(({ numPages }) => {
    setNumPages(numPages)
  }, [])

  // View article/knowledge entry
  const handleViewArticle = (entry) => {
    setCurrentArticle(entry)
    setArticleViewerVisible(true)
  }

  // Get entry type tag color
  const getTypeColor = (type) => {
    const colors = {
      'SUCCESS_PATTERN': 'green',
      'FAILURE_PITFALL': 'red',
      'FIELD_BLACKLIST': 'orange',
      'OPERATOR_STAT': 'blue',
      'FIELD_INSIGHT': 'purple',
    }
    return colors[type] || 'default'
  }

  // Get source tag
  const getSourceTag = (entry) => {
    const source = entry.meta_data?.source || entry.created_by
    const sourceColors = {
      'forum': 'cyan',
      'paper': 'gold',
      'documentation': 'purple',
      'FORUM_SYNC': 'cyan',
      'PAPER_IMPORT': 'gold',
      'USER_EXTERNAL': 'blue',
      'SYSTEM': 'green',
      'MINING_AGENT': 'green',
      'USER': 'blue',
    }
    const sourceLabels = {
      'forum': '论坛',
      'paper': '论文',
      'documentation': '文档',
      'FORUM_SYNC': '论坛同步',
      'PAPER_IMPORT': '论文',
      'USER_EXTERNAL': '手动添加',
      'SYSTEM': '系统',
      'MINING_AGENT': '挖掘',
      'USER': '用户',
    }
    return (
      <Tag color={sourceColors[source] || 'default'}>
        {sourceLabels[source] || source}
      </Tag>
    )
  }

  // Knowledge table columns
  const knowledgeColumns = [
    {
      title: '类型',
      dataIndex: 'entry_type',
      key: 'entry_type',
      width: 120,
      render: (type) => <Tag color={getTypeColor(type)}>{type}</Tag>,
    },
    {
      title: '模式/内容',
      dataIndex: 'pattern',
      key: 'pattern',
      ellipsis: true,
      render: (pattern, record) => (
        <Tooltip title={pattern}>
          <Text code style={{ fontSize: 12 }}>
            {pattern?.length > 80 ? pattern.substring(0, 80) + '...' : pattern}
          </Text>
        </Tooltip>
      ),
    },
    {
      title: '描述',
      dataIndex: 'description',
      key: 'description',
      ellipsis: true,
      width: 200,
    },
    {
      title: '来源',
      key: 'source',
      width: 100,
      render: (_, record) => getSourceTag(record),
    },
    {
      title: '使用次数',
      dataIndex: 'usage_count',
      key: 'usage_count',
      width: 80,
      sorter: (a, b) => a.usage_count - b.usage_count,
    },
    {
      title: '状态',
      dataIndex: 'is_active',
      key: 'is_active',
      width: 70,
      render: (active) => (
        <Badge status={active ? 'success' : 'default'} text={active ? '活跃' : '禁用'} />
      ),
    },
    {
      title: '操作',
      key: 'actions',
      width: 150,
      render: (_, record) => (
        <Space size="small">
          <Tooltip title="查看详情">
            <Button 
              type="text" 
              size="small" 
              icon={<EyeOutlined />} 
              onClick={() => handleViewArticle(record)}
            />
          </Tooltip>
          <Tooltip title="编辑">
            <Button 
              type="text" 
              size="small" 
              icon={<EditOutlined />} 
              onClick={() => handleEdit(record)}
            />
          </Tooltip>
          <Popconfirm
            title="确定删除此知识条目？"
            onConfirm={() => deleteMutation.mutate(record.id)}
            okText="确定"
            cancelText="取消"
          >
            <Tooltip title="删除">
              <Button 
                type="text" 
                size="small" 
                danger
                icon={<DeleteOutlined />} 
              />
            </Tooltip>
          </Popconfirm>
          {record.meta_data?.source_url && (
            <Tooltip title="查看来源">
              <Button 
                type="text" 
                size="small" 
                icon={<LinkOutlined />} 
                onClick={() => window.open(record.meta_data.source_url, '_blank')}
              />
            </Tooltip>
          )}
        </Space>
      ),
    },
  ]

  // Stats Card
  const StatsCard = () => (
    <Row gutter={16} style={{ marginBottom: 24 }}>
      <Col xs={12} sm={6}>
        <Card className="glass-card">
          <Statistic 
            title="知识总数" 
            value={statsData?.total_entries || 0} 
            prefix={<BookOutlined style={{ color: '#00d4ff' }} />}
          />
        </Card>
      </Col>
      <Col xs={12} sm={6}>
        <Card className="glass-card">
          <Statistic 
            title="外部知识" 
            value={statsData?.external_count || 0} 
            prefix={<GlobalOutlined style={{ color: '#52c41a' }} />}
          />
        </Card>
      </Col>
      <Col xs={12} sm={6}>
        <Card className="glass-card">
          <Statistic 
            title="系统知识" 
            value={statsData?.system_count || 0} 
            prefix={<RobotOutlined style={{ color: '#722ed1' }} />}
          />
        </Card>
      </Col>
      <Col xs={12} sm={6}>
        <Card className="glass-card">
          <Statistic 
            title="本周新增" 
            value={statsData?.recent_entries || 0} 
            prefix={<ExperimentOutlined style={{ color: '#faad14' }} />}
          />
        </Card>
      </Col>
    </Row>
  )

  // External Knowledge Tab
  const ExternalKnowledgeTab = () => (
    <div>
      <Row gutter={16} style={{ marginBottom: 16 }}>
        <Col flex="auto">
          <Space>
            <Button 
              type="primary" 
              icon={<PlusOutlined />}
              onClick={() => {
                addForm.setFieldsValue({ source_type: 'manual' })
                setAddModalVisible(true)
              }}
            >
              添加知识
            </Button>
            <Button 
              icon={<SyncOutlined />}
              onClick={() => forumSyncMutation.mutate({ search_terms: ['high sharpe', 'momentum'], max_posts: 30 })}
              loading={forumSyncMutation.isPending}
            >
              同步论坛
            </Button>
            <Button 
              icon={<FileTextOutlined />}
              onClick={() => setPaperModalVisible(true)}
            >
              添加论文
            </Button>
          </Space>
        </Col>
        <Col>
          <Button icon={<SyncOutlined />} onClick={() => refetchExternal()}>
            刷新
          </Button>
        </Col>
      </Row>

      {externalLoading ? (
        <Spin size="large" />
      ) : (
        <Table
          dataSource={externalKnowledge || []}
          columns={knowledgeColumns}
          rowKey="id"
          pagination={{ pageSize: 15 }}
          size="small"
        />
      )}
    </div>
  )

  // System Knowledge Tab
  const SystemKnowledgeTab = () => (
    <div>
      <Row gutter={16} style={{ marginBottom: 16 }}>
        <Col flex="auto">
          <Alert
            message="系统自动积累的挖掘知识"
            description="这些知识来自于系统的 Alpha 挖掘过程，包括成功模式、失败教训、字段洞察等。"
            type="info"
            showIcon
          />
        </Col>
        <Col>
          <Button icon={<SyncOutlined />} onClick={() => refetchSystem()}>
            刷新
          </Button>
        </Col>
      </Row>

      {systemLoading ? (
        <Spin size="large" />
      ) : (
        <Table
          dataSource={systemKnowledge || []}
          columns={knowledgeColumns}
          rowKey="id"
          pagination={{ pageSize: 15 }}
          size="small"
        />
      )}
    </div>
  )

  // Forum Search Tab
  const ForumSearchTab = () => (
    <div>
      <Card className="glass-card" style={{ marginBottom: 16 }}>
        <Space.Compact style={{ width: '100%' }}>
          <Input
            placeholder="搜索 WorldQuant BRAIN 论坛..."
            value={forumSearchQuery}
            onChange={(e) => setForumSearchQuery(e.target.value)}
            onPressEnter={handleForumSearch}
            prefix={<SearchOutlined />}
            style={{ flex: 1 }}
          />
          <Button 
            type="primary" 
            icon={<SearchOutlined />}
            onClick={handleForumSearch}
            loading={forumLoading}
          >
            搜索
          </Button>
        </Space.Compact>
        <Text type="secondary" style={{ display: 'block', marginTop: 8 }}>
          提示：搜索论坛中的 Alpha 技巧、策略讨论等内容
        </Text>
      </Card>

      {forumLoading ? (
        <Spin size="large" />
      ) : forumPosts.length > 0 ? (
        <Row gutter={16}>
          {forumPosts.map((post, index) => (
            <Col xs={24} lg={12} key={index} style={{ marginBottom: 16 }}>
              <Card 
                className="glass-card"
                title={
                  <Space>
                    <Text strong ellipsis style={{ maxWidth: 300 }}>
                      {post.title}
                    </Text>
                    <Tag color="blue">{post.likes} 赞</Tag>
                  </Space>
                }
                extra={
                  <Space>
                    <Button 
                      size="small"
                      icon={<EyeOutlined />}
                      onClick={() => handleViewArticle({
                        id: post.post_id,
                        entry_type: 'FORUM_POST',
                        pattern: post.alpha_patterns?.join('\n') || post.title,
                        description: post.content,
                        meta_data: {
                          source: 'forum',
                          source_title: post.title,
                          author: post.author,
                          likes: post.likes,
                          views: post.views,
                          url: post.url,
                        },
                        usage_count: 0,
                        is_active: true,
                        created_by: 'FORUM',
                        created_at: new Date().toISOString(),
                      })}
                    >
                      查看
                    </Button>
                    <Button 
                      type="primary" 
                      size="small"
                      icon={<CloudDownloadOutlined />}
                      onClick={() => importForumMutation.mutate(post)}
                      loading={importForumMutation.isPending}
                    >
                      导入
                    </Button>
                  </Space>
                }
              >
                <Paragraph ellipsis={{ rows: 3 }}>
                  {post.content}
                </Paragraph>
                
                {post.alpha_patterns?.length > 0 && (
                  <div style={{ marginTop: 8 }}>
                    <Text type="secondary">发现的 Alpha 模式:</Text>
                    <div style={{ marginTop: 4 }}>
                      {post.alpha_patterns.slice(0, 2).map((pattern, i) => (
                        <Tag key={i} color="green" style={{ marginBottom: 4 }}>
                          {pattern.length > 50 ? pattern.substring(0, 50) + '...' : pattern}
                        </Tag>
                      ))}
                    </div>
                  </div>
                )}
                
                <Divider style={{ margin: '12px 0' }} />
                
                <Space>
                  <Text type="secondary">作者: {post.author}</Text>
                  <Text type="secondary">相关度: {(post.relevance_score * 100).toFixed(0)}%</Text>
                </Space>
              </Card>
            </Col>
          ))}
        </Row>
      ) : (
        <Empty description="输入关键词搜索论坛内容" />
      )}
    </div>
  )

  // Papers Tab
  const PapersTab = () => (
    <div>
      <Row gutter={16} style={{ marginBottom: 16 }}>
        <Col flex="auto">
          <Button 
            type="primary" 
            icon={<PlusOutlined />}
            onClick={() => setPaperModalVisible(true)}
          >
            添加论文
          </Button>
        </Col>
        <Col>
          <Button icon={<SyncOutlined />} onClick={() => refetchPapers()}>
            刷新
          </Button>
        </Col>
      </Row>

      {papersLoading ? (
        <Spin size="large" />
      ) : (
        <Row gutter={16}>
          {(papersData || []).map((paper, index) => (
            <Col xs={24} md={12} lg={8} key={index} style={{ marginBottom: 16 }}>
              <Card 
                className="glass-card"
                title={
                  <Tooltip title={paper.title}>
                    <Text strong ellipsis style={{ maxWidth: 250 }}>
                      {paper.title}
                    </Text>
                  </Tooltip>
                }
                extra={
                  paper.downloaded ? (
                    <Tag color="green">已下载</Tag>
                  ) : (
                    <Tag color="orange">仅链接</Tag>
                  )
                }
              >
                <Paragraph ellipsis={{ rows: 2 }} type="secondary">
                  {paper.description}
                </Paragraph>
                
                <Statistic 
                  title="提取模式数" 
                  value={paper.patterns_count} 
                  style={{ marginBottom: 12 }}
                />
                
                <Space wrap>
                  {paper.downloaded && (
                    <Button 
                      size="small" 
                      type="primary"
                      icon={<EyeOutlined />}
                      onClick={() => handleViewPdf(paper)}
                    >
                      在线查看
                    </Button>
                  )}
                  {paper.source_url && (
                    <Button 
                      size="small" 
                      icon={<LinkOutlined />}
                      onClick={() => window.open(paper.source_url, '_blank')}
                    >
                      原文链接
                    </Button>
                  )}
                  {paper.downloaded && (
                    <Button 
                      size="small" 
                      icon={<DownloadOutlined />}
                      onClick={() => window.open(`/api/v1/knowledge/papers/${paper.id}/download`, '_blank')}
                    >
                      下载
                    </Button>
                  )}
                </Space>
              </Card>
            </Col>
          ))}
          
          {(!papersData || papersData.length === 0) && (
            <Col span={24}>
              <Empty description="暂无论文，点击上方按钮添加" />
            </Col>
          )}
        </Row>
      )}
    </div>
  )

  const tabs = [
    {
      key: 'external',
      label: (
        <Space>
          <GlobalOutlined />
          外部知识
          <Badge count={externalKnowledge?.length || 0} style={{ marginLeft: 4 }} />
        </Space>
      ),
      children: <ExternalKnowledgeTab />,
    },
    {
      key: 'system',
      label: (
        <Space>
          <RobotOutlined />
          系统知识
          <Badge count={systemKnowledge?.length || 0} style={{ marginLeft: 4 }} />
        </Space>
      ),
      children: <SystemKnowledgeTab />,
    },
    {
      key: 'forum',
      label: (
        <Space>
          <SearchOutlined />
          论坛搜索
        </Space>
      ),
      children: <ForumSearchTab />,
    },
    {
      key: 'papers',
      label: (
        <Space>
          <FileTextOutlined />
          学术论文
          <Badge count={papersData?.length || 0} style={{ marginLeft: 4 }} />
        </Space>
      ),
      children: <PapersTab />,
    },
  ]

  return (
    <div>
      <Title level={3} style={{ marginBottom: 24 }}>
        <BookOutlined style={{ marginRight: 12, color: '#00d4ff' }} />
        知识库管理
      </Title>

      <StatsCard />

      <Card className="glass-card">
        <Tabs items={tabs} size="large" defaultActiveKey="external" />
      </Card>

      {/* Add Knowledge Modal */}
      <Modal
        title="添加知识条目"
        open={addModalVisible}
        onCancel={() => setAddModalVisible(false)}
        footer={null}
        width={600}
      >
        <Form
          form={addForm}
          layout="vertical"
          onFinish={(values) => {
            createMutation.mutate({
              entry_type: values.entry_type,
              pattern: values.pattern,
              description: values.description,
              meta_data: { source: values.source_type },
              source_type: values.source_type,
              source_url: values.source_url,
            })
          }}
        >
          <Form.Item
            name="entry_type"
            label="知识类型"
            rules={[{ required: true, message: '请选择类型' }]}
          >
            <Select placeholder="选择类型">
              <Option value="SUCCESS_PATTERN">成功模式</Option>
              <Option value="FAILURE_PITFALL">失败教训</Option>
              <Option value="FIELD_INSIGHT">字段洞察</Option>
              <Option value="FIELD_BLACKLIST">字段黑名单</Option>
            </Select>
          </Form.Item>

          <Form.Item
            name="source_type"
            label="来源类型"
            initialValue="manual"
          >
            <Select>
              <Option value="manual">手动添加</Option>
              <Option value="forum">论坛</Option>
              <Option value="paper">论文</Option>
              <Option value="documentation">文档</Option>
            </Select>
          </Form.Item>

          <Form.Item name="source_url" label="来源链接 (可选)">
            <Input placeholder="https://..." />
          </Form.Item>

          <Form.Item
            name="pattern"
            label="模式/表达式"
            rules={[{ required: true, message: '请输入内容' }]}
          >
            <TextArea rows={3} placeholder="输入 Alpha 表达式或知识模式" />
          </Form.Item>

          <Form.Item
            name="description"
            label="描述"
          >
            <TextArea rows={2} placeholder="描述这个知识的用途和背景" />
          </Form.Item>

          <Form.Item>
            <Space>
              <Button type="primary" htmlType="submit" loading={createMutation.isPending}>
                创建
              </Button>
              <Button onClick={() => setAddModalVisible(false)}>
                取消
              </Button>
            </Space>
          </Form.Item>
        </Form>
      </Modal>

      {/* Edit Knowledge Modal */}
      <Modal
        title="编辑知识条目"
        open={editModalVisible}
        onCancel={() => {
          setEditModalVisible(false)
          setEditingEntry(null)
        }}
        footer={null}
        width={600}
      >
        <Form
          form={editForm}
          layout="vertical"
          onFinish={(values) => {
            updateMutation.mutate({
              id: editingEntry.id,
              data: values,
            })
          }}
        >
          <Form.Item
            name="pattern"
            label="模式/表达式"
          >
            <TextArea rows={3} />
          </Form.Item>

          <Form.Item
            name="description"
            label="描述"
          >
            <TextArea rows={2} />
          </Form.Item>

          <Form.Item
            name="is_active"
            label="状态"
          >
            <Select>
              <Option value={true}>活跃</Option>
              <Option value={false}>禁用</Option>
            </Select>
          </Form.Item>

          <Form.Item>
            <Space>
              <Button type="primary" htmlType="submit" loading={updateMutation.isPending}>
                保存
              </Button>
              <Button onClick={() => setEditModalVisible(false)}>
                取消
              </Button>
            </Space>
          </Form.Item>
        </Form>
      </Modal>

      {/* Add Paper Modal */}
      <Modal
        title="添加学术论文"
        open={paperModalVisible}
        onCancel={() => setPaperModalVisible(false)}
        footer={null}
        width={600}
      >
        <Alert
          message="支持的论文来源"
          description="arXiv (自动下载 PDF), SSRN, 或直接 PDF 链接"
          type="info"
          showIcon
          style={{ marginBottom: 16 }}
        />
        
        <Form
          form={paperForm}
          layout="vertical"
          onFinish={(values) => {
            downloadPaperMutation.mutate(values)
          }}
        >
          <Form.Item
            name="url"
            label="论文链接"
            rules={[{ required: true, message: '请输入链接' }]}
          >
            <Input placeholder="https://arxiv.org/abs/..." />
          </Form.Item>

          <Form.Item
            name="title"
            label="论文标题"
            rules={[{ required: true, message: '请输入标题' }]}
          >
            <Input placeholder="例如: 101 Formulaic Alphas" />
          </Form.Item>

          <Form.Item
            name="description"
            label="论文描述"
          >
            <TextArea rows={2} placeholder="简要描述论文内容" />
          </Form.Item>

          <Form.Item>
            <Space>
              <Button 
                type="primary" 
                htmlType="submit" 
                icon={<CloudDownloadOutlined />}
                loading={downloadPaperMutation.isPending}
              >
                下载并添加
              </Button>
              <Button onClick={() => setPaperModalVisible(false)}>
                取消
              </Button>
            </Space>
          </Form.Item>
        </Form>
      </Modal>

      {/* PDF Viewer Modal */}
      <Modal
        title={
          <Space>
            <FilePdfOutlined style={{ color: '#ff4d4f' }} />
            <span>{currentPdf?.title || '论文预览'}</span>
            {numPages && (
              <Tag color="blue">共 {numPages} 页</Tag>
            )}
          </Space>
        }
        open={pdfViewerVisible}
        onCancel={() => {
          setPdfViewerVisible(false)
          setCurrentPdf(null)
          setNumPages(null)
          setCurrentPage(1)
        }}
        footer={
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            {/* 分页控制 */}
            <Space>
              <Button 
                icon={<LeftOutlined />}
                disabled={currentPage <= 1}
                onClick={() => setCurrentPage(p => Math.max(1, p - 1))}
              >
                上一页
              </Button>
              <Text>
                第 {currentPage} / {numPages || '?'} 页
              </Text>
              <Button 
                icon={<RightOutlined />}
                disabled={currentPage >= numPages}
                onClick={() => setCurrentPage(p => Math.min(numPages || p, p + 1))}
              >
                下一页
              </Button>
            </Space>
            
            {/* 缩放控制 */}
            <Space>
              <Button 
                icon={<ZoomOutOutlined />}
                onClick={() => setPdfScale(s => Math.max(0.5, s - 0.2))}
                disabled={pdfScale <= 0.5}
              />
              <Text>{Math.round(pdfScale * 100)}%</Text>
              <Button 
                icon={<ZoomInOutlined />}
                onClick={() => setPdfScale(s => Math.min(2.5, s + 0.2))}
                disabled={pdfScale >= 2.5}
              />
            </Space>
            
            {/* 操作按钮 */}
            <Space>
              <Button 
                icon={<FullscreenOutlined />}
                onClick={() => window.open(`/api/v1/knowledge/papers/${currentPdf?.id}/download`, '_blank')}
              >
                新窗口
              </Button>
              <Button 
                icon={<DownloadOutlined />}
                onClick={() => {
                  const link = document.createElement('a')
                  link.href = `/api/v1/knowledge/papers/${currentPdf?.id}/download`
                  link.download = `${currentPdf?.title || 'paper'}.pdf`
                  link.click()
                }}
              >
                下载
              </Button>
              <Button onClick={() => setPdfViewerVisible(false)}>
                关闭
              </Button>
            </Space>
          </div>
        }
        width="90%"
        style={{ top: 20 }}
        styles={{ 
          body: { 
            height: 'calc(100vh - 220px)', 
            overflow: 'auto',
            display: 'flex',
            justifyContent: 'center',
            background: '#525659',
            padding: 16,
          } 
        }}
      >
        {currentPdf && (
          <Document
            file={`/api/v1/knowledge/papers/${currentPdf.id}/download`}
            onLoadSuccess={onPdfLoadSuccess}
            loading={
              <div style={{ 
                display: 'flex', 
                flexDirection: 'column',
                alignItems: 'center', 
                justifyContent: 'center',
                height: 400,
                color: '#fff'
              }}>
                <Spin size="large" />
                <Text style={{ color: '#fff', marginTop: 16 }}>正在加载论文...</Text>
              </div>
            }
            error={
              <div style={{ 
                display: 'flex', 
                flexDirection: 'column',
                alignItems: 'center', 
                justifyContent: 'center',
                height: 400,
                color: '#fff'
              }}>
                <Alert
                  message="PDF 加载失败"
                  description="无法渲染 PDF 文件，请尝试在新窗口中打开"
                  type="error"
                  showIcon
                />
                <Button 
                  type="primary"
                  style={{ marginTop: 16 }}
                  onClick={() => window.open(`/api/v1/knowledge/papers/${currentPdf?.id}/download`, '_blank')}
                >
                  在新窗口打开
                </Button>
              </div>
            }
          >
            <Page 
              pageNumber={currentPage} 
              scale={pdfScale}
              renderTextLayer={true}
              renderAnnotationLayer={true}
            />
          </Document>
        )}
      </Modal>

      {/* Article/Knowledge Entry Viewer Modal */}
      <Modal
        title={
          <Space>
            <ReadOutlined style={{ color: '#00d4ff' }} />
            <span>知识详情</span>
          </Space>
        }
        open={articleViewerVisible}
        onCancel={() => {
          setArticleViewerVisible(false)
          setCurrentArticle(null)
        }}
        footer={
          <Space>
            {currentArticle?.meta_data?.source_url && (
              <Button 
                icon={<LinkOutlined />}
                onClick={() => window.open(currentArticle.meta_data.source_url, '_blank')}
              >
                查看原文
              </Button>
            )}
            <Button 
              icon={<EditOutlined />}
              onClick={() => {
                setArticleViewerVisible(false)
                handleEdit(currentArticle)
              }}
            >
              编辑
            </Button>
            <Button onClick={() => setArticleViewerVisible(false)}>
              关闭
            </Button>
          </Space>
        }
        width={800}
      >
        {currentArticle && (
          <div>
            {/* Header Info */}
            <Row gutter={16} style={{ marginBottom: 16 }}>
              <Col span={12}>
                <Text type="secondary">类型：</Text>
                <Tag color={getTypeColor(currentArticle.entry_type)}>
                  {currentArticle.entry_type}
                </Tag>
              </Col>
              <Col span={12}>
                <Text type="secondary">来源：</Text>
                {getSourceTag(currentArticle)}
              </Col>
            </Row>
            
            <Row gutter={16} style={{ marginBottom: 16 }}>
              <Col span={12}>
                <Text type="secondary">使用次数：</Text>
                <Text strong>{currentArticle.usage_count}</Text>
              </Col>
              <Col span={12}>
                <Text type="secondary">状态：</Text>
                <Badge 
                  status={currentArticle.is_active ? 'success' : 'default'} 
                  text={currentArticle.is_active ? '活跃' : '禁用'} 
                />
              </Col>
            </Row>

            <Divider />
            
            {/* Pattern/Expression */}
            <div style={{ marginBottom: 16 }}>
              <Title level={5}>
                <FileTextOutlined style={{ marginRight: 8 }} />
                模式/表达式
              </Title>
              <Card 
                size="small" 
                style={{ 
                  background: 'rgba(0, 212, 255, 0.1)', 
                  border: '1px solid rgba(0, 212, 255, 0.3)' 
                }}
              >
                <Text code style={{ 
                  fontSize: 13, 
                  whiteSpace: 'pre-wrap', 
                  wordBreak: 'break-all' 
                }}>
                  {currentArticle.pattern}
                </Text>
              </Card>
            </div>
            
            {/* Description */}
            {currentArticle.description && (
              <div style={{ marginBottom: 16 }}>
                <Title level={5}>
                  <BookOutlined style={{ marginRight: 8 }} />
                  描述
                </Title>
                <Paragraph style={{ marginBottom: 0 }}>
                  {currentArticle.description}
                </Paragraph>
              </div>
            )}
            
            {/* Metadata */}
            {currentArticle.meta_data && Object.keys(currentArticle.meta_data).length > 0 && (
              <div style={{ marginBottom: 16 }}>
                <Title level={5}>
                  <ExperimentOutlined style={{ marginRight: 8 }} />
                  元数据
                </Title>
                <Card size="small">
                  {Object.entries(currentArticle.meta_data).map(([key, value]) => (
                    <div key={key} style={{ marginBottom: 4 }}>
                      <Text type="secondary">{key}: </Text>
                      <Text>
                        {typeof value === 'object' 
                          ? JSON.stringify(value) 
                          : String(value)
                        }
                      </Text>
                    </div>
                  ))}
                </Card>
              </div>
            )}
            
            {/* Timestamps */}
            <Divider />
            <Row gutter={16}>
              <Col span={12}>
                <Text type="secondary">创建时间：</Text>
                <Text>{new Date(currentArticle.created_at).toLocaleString()}</Text>
              </Col>
              <Col span={12}>
                <Text type="secondary">更新时间：</Text>
                <Text>
                  {currentArticle.updated_at 
                    ? new Date(currentArticle.updated_at).toLocaleString() 
                    : '-'
                  }
                </Text>
              </Col>
            </Row>
          </div>
        )}
      </Modal>
    </div>
  )
}
