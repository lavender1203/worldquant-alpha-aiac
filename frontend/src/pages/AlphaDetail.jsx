import { useParams, useNavigate } from 'react-router-dom'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { 
  Row, 
  Col, 
  Card, 
  Typography, 
  Tag, 
  Button, 
  Space, 
  Descriptions,
  Spin,
  Empty,
  Input,
  message,
  Divider,
} from 'antd'
import {
  ArrowLeftOutlined,
  LikeOutlined,
  DislikeOutlined,
  CopyOutlined,
} from '@ant-design/icons'
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer 
} from 'recharts'
import api from '../services/api'

const { Title, Text, Paragraph } = Typography
const { TextArea } = Input

// Mock PnL data for demo
const mockPnL = [
  { date: '2025-01', returns: 0 },
  { date: '2025-02', returns: 1.5 },
  { date: '2025-03', returns: 3.2 },
  { date: '2025-04', returns: 2.8 },
  { date: '2025-05', returns: 5.1 },
  { date: '2025-06', returns: 6.8 },
  { date: '2025-07', returns: 8.2 },
  { date: '2025-08', returns: 7.5 },
  { date: '2025-09', returns: 9.1 },
  { date: '2025-10', returns: 11.3 },
  { date: '2025-11', returns: 10.8 },
  { date: '2025-12', returns: 12.5 },
]

export default function AlphaDetail() {
  const { id } = useParams()
  const navigate = useNavigate()
  const queryClient = useQueryClient()

  // Fetch alpha details
  const { data: alpha, isLoading } = useQuery({
    queryKey: ['alpha', id],
    queryFn: () => api.getAlpha(id),
  })

  // Feedback mutation
  const feedbackMutation = useMutation({
    mutationFn: ({ rating, comment }) => api.submitAlphaFeedback(id, rating, comment),
    onSuccess: () => {
      message.success('反馈已提交')
      queryClient.invalidateQueries(['alpha', id])
    },
  })

  const handleFeedback = (rating) => {
    feedbackMutation.mutate({ rating })
  }

  const copyExpression = () => {
    navigator.clipboard.writeText(alpha.expression)
    message.success('表达式已复制到剪贴板')
  }

  if (isLoading) {
    return (
      <div style={{ textAlign: 'center', padding: 100 }}>
        <Spin size="large" />
      </div>
    )
  }

  if (!alpha) {
    return (
      <Empty description="未找到 Alpha">
        <Button onClick={() => navigate('/alphas')}>返回实验室</Button>
      </Empty>
    )
  }

  const metrics = alpha.metrics || {}

  return (
    <div>
      {/* Header */}
      <Row justify="space-between" align="middle" style={{ marginBottom: 24 }}>
        <Col>
          <Space>
            <Button icon={<ArrowLeftOutlined />} onClick={() => navigate('/alphas')}>
              返回
            </Button>
            <Title level={3} style={{ margin: 0 }}>
              Alpha #{alpha.id}
            </Title>
            <Tag
              color={
                alpha.quality_status === 'PASS'
                  ? 'success'
                  : (alpha.quality_status === 'PROMISING'
                    ? 'processing'
                    : (alpha.quality_status === 'OPTIMIZE'
                      ? 'warning'
                      : (alpha.quality_status === 'FAIL' ? 'error' : 'default')))
              }
            >
              {alpha.quality_status}
            </Tag>
          </Space>
        </Col>
      </Row>

      <Row gutter={[16, 16]}>
        {/* Left: Expression & Info */}
        <Col xs={24} lg={14}>
          {/* Expression Card */}
          <Card 
            className="glass-card" 
            title="表达式"
            extra={
              <Button icon={<CopyOutlined />} size="small" onClick={copyExpression}>
                复制
              </Button>
            }
          >
            <pre style={{ 
              fontSize: 14,
              lineHeight: 1.6,
              overflow: 'auto',
              maxHeight: 200,
            }}>
              {alpha.expression}
            </pre>
          </Card>

          {/* Hypothesis & Explanation */}
          {(alpha.hypothesis || alpha.logic_explanation) && (
            <Card className="glass-card" title="分析" style={{ marginTop: 16 }}>
              {alpha.hypothesis && (
                <>
                  <Text strong>假设 (Hypothesis):</Text>
                  <Paragraph style={{ color: 'rgba(255,255,255,0.85)' }}>
                    {alpha.hypothesis}
                  </Paragraph>
                </>
              )}
              {alpha.logic_explanation && (
                <>
                  <Text strong>逻辑解释:</Text>
                  <Paragraph style={{ color: 'rgba(255,255,255,0.85)' }}>
                    {alpha.logic_explanation}
                  </Paragraph>
                </>
              )}
            </Card>
          )}

          {/* PnL Chart */}
          <Card className="glass-card" title="累计收益" style={{ marginTop: 16 }}>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={mockPnL}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                <XAxis dataKey="date" stroke="rgba(255,255,255,0.5)" />
                <YAxis stroke="rgba(255,255,255,0.5)" unit="%" />
                <Tooltip 
                  contentStyle={{ 
                    background: '#131a2b', 
                    border: '1px solid rgba(255,255,255,0.1)',
                    borderRadius: 8,
                  }}
                />
                <Line 
                  type="monotone" 
                  dataKey="returns" 
                  stroke="#00ff88" 
                  strokeWidth={2}
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </Card>
        </Col>

        {/* Right: Metrics & Feedback */}
        <Col xs={24} lg={10}>
          {/* Metrics */}
          <Card className="glass-card" title="绩效指标">
            <Descriptions column={1} size="small">
              <Descriptions.Item label="夏普比率">
                <Text style={{ 
                  fontSize: 18, 
                  fontWeight: 600,
                  color: metrics.sharpe >= 1.5 ? '#00ff88' : '#ffb700'
                }}>
                  {metrics.sharpe?.toFixed(2) || '--'}
                </Text>
              </Descriptions.Item>
              <Descriptions.Item label="收益率">
                {metrics.returns?.toFixed(2)}%
              </Descriptions.Item>
              <Descriptions.Item label="换手率">
                {metrics.turnover?.toFixed(2) || '--'}
              </Descriptions.Item>
              <Descriptions.Item label="最大回撤">
                {metrics.max_dd?.toFixed(2)}%
              </Descriptions.Item>
              <Descriptions.Item label="Fitness">
                {metrics.fitness?.toFixed(2) || '--'}
              </Descriptions.Item>
            </Descriptions>
          </Card>

          {/* Metadata */}
          <Card className="glass-card" title="元数据" style={{ marginTop: 16 }}>
            <Descriptions column={1} size="small">
              <Descriptions.Item label="地区">{alpha.region}</Descriptions.Item>
              <Descriptions.Item label="股票池">{alpha.universe}</Descriptions.Item>
              <Descriptions.Item label="数据集">{alpha.dataset_id}</Descriptions.Item>
              <Descriptions.Item label="使用字段">
                <Space wrap>
                  {(alpha.fields_used || []).map(f => (
                    <Tag key={f} size="small">{f}</Tag>
                  ))}
                </Space>
              </Descriptions.Item>
              <Descriptions.Item label="使用算子">
                <Space wrap>
                  {(alpha.operators_used || []).map(o => (
                    <Tag key={o} size="small" color="blue">{o}</Tag>
                  ))}
                </Space>
              </Descriptions.Item>
              <Descriptions.Item label="创建时间">
                {new Date(alpha.created_at).toLocaleString()}
              </Descriptions.Item>
            </Descriptions>
          </Card>

          {/* Human Feedback */}
          <Card className="glass-card" title="人工反馈" style={{ marginTop: 16 }}>
            <div style={{ marginBottom: 16 }}>
              <Text>当前评价: </Text>
              {alpha.human_feedback === 'LIKED' && (
                <Tag icon={<LikeOutlined />} color="success">喜欢</Tag>
              )}
              {alpha.human_feedback === 'DISLIKED' && (
                <Tag icon={<DislikeOutlined />} color="error">不喜欢</Tag>
              )}
              {alpha.human_feedback === 'NONE' && (
                <Text type="secondary">未评价</Text>
              )}
            </div>

            <Space>
              <Button 
                icon={<LikeOutlined />} 
                type={alpha.human_feedback === 'LIKED' ? 'primary' : 'default'}
                onClick={() => handleFeedback('LIKED')}
                loading={feedbackMutation.isLoading}
              >
                点赞
              </Button>
              <Button 
                icon={<DislikeOutlined />} 
                danger={alpha.human_feedback === 'DISLIKED'}
                onClick={() => handleFeedback('DISLIKED')}
                loading={feedbackMutation.isLoading}
              >
                踩
              </Button>
            </Space>

            {alpha.feedback_comment && (
              <>
                <Divider />
                <Text strong>评论:</Text>
                <Paragraph style={{ marginTop: 8 }}>
                  {alpha.feedback_comment}
                </Paragraph>
              </>
            )}
          </Card>
        </Col>
      </Row>
    </div>
  )
}
