import React from 'react'
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
  Table,
  Timeline,
  Collapse,
  Spin,
  Empty,
  Select,
  message,
} from 'antd'
import {
  ArrowLeftOutlined,
  PlayCircleOutlined,
  PauseCircleOutlined,
  StopOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  LoadingOutlined,
  SearchOutlined,
  BulbOutlined,
  CodeOutlined,
  ExperimentOutlined,
  SyncOutlined,
  ReloadOutlined,
} from '@ant-design/icons'
import api from '../services/api'

const { Title, Text, Paragraph } = Typography

// Step type icons
const stepIcons = {
  RAG_QUERY: <SearchOutlined />,
  HYPOTHESIS: <BulbOutlined />,
  CODE_GEN: <CodeOutlined />,
  VALIDATE: <CheckCircleOutlined />,
  SIMULATE: <ExperimentOutlined />,
  SELF_CORRECT: <SyncOutlined />,
  EVALUATE: <CheckCircleOutlined />,
}

// Step status colors
const statusColors = {
  SUCCESS: 'green',
  FAILED: 'red',
  RUNNING: 'processing',
  SKIPPED: 'default',
}

const asArray = (value) => Array.isArray(value) ? value : []

export default function TaskDetail() {
  const { id } = useParams()
  const navigate = useNavigate()

  const [selectedRunId, setSelectedRunId] = React.useState(null)

  // Fetch task details with trace
  const { data: task, isLoading, error } = useQuery({
    queryKey: ['task', id],
    queryFn: () => api.getTask(id),
    refetchInterval: 5000, // Refresh while task is running
  })

  const { data: runs } = useQuery({
    queryKey: ['taskRuns', id],
    queryFn: () => api.getTaskRuns(id),
    enabled: !!id,
    refetchInterval: task?.status === 'RUNNING' ? 5000 : false,
  })

  React.useEffect(() => {
    if (!runs || runs.length === 0) return
    if (selectedRunId == null) {
      setSelectedRunId(runs[0].id)
    }
  }, [runs, selectedRunId])

  const { data: runTrace } = useQuery({
    queryKey: ['runTrace', selectedRunId],
    queryFn: () => api.getRunTrace(selectedRunId),
    enabled: !!selectedRunId,
    refetchInterval: task?.status === 'RUNNING' ? 5000 : false,
  })

  const { data: runAlphasResp, isLoading: isRunAlphasLoading } = useQuery({
    queryKey: ['runAlphas', selectedRunId],
    queryFn: () => api.getRunAlphas(selectedRunId, { limit: 200, offset: 0 }),
    enabled: !!selectedRunId,
    refetchInterval: task?.status === 'RUNNING' ? 5000 : false,
  })

  const runAlphas = React.useMemo(() => runAlphasResp?.items || [], [runAlphasResp])
  const selectedRun = React.useMemo(
    () => (runs || []).find(r => r.id === selectedRunId),
    [runs, selectedRunId]
  )

  const runAlphaSummary = React.useMemo(() => {
    const counts = { PASS: 0, OPTIMIZE: 0, FAIL: 0, OTHER: 0 }
    const scores = []
    const sharpes = []

    for (const a of runAlphas) {
      const status = a.quality_status || 'OTHER'
      if (status === 'PASS') counts.PASS += 1
      else if (status === 'OPTIMIZE') counts.OPTIMIZE += 1
      else if (status === 'FAIL') counts.FAIL += 1
      else counts.OTHER += 1

      const s = a.metrics?._score
      if (typeof s === 'number') scores.push(s)

      const sh = a.metrics?.sharpe
      if (typeof sh === 'number') sharpes.push(sh)
    }

    const avgScore = scores.length ? scores.reduce((x, y) => x + y, 0) / scores.length : null
    const bestScore = scores.length ? Math.max(...scores) : null
    const avgSharpe = sharpes.length ? sharpes.reduce((x, y) => x + y, 0) / sharpes.length : null
    const bestSharpe = sharpes.length ? Math.max(...sharpes) : null

    const top = [...runAlphas]
      .filter(a => typeof a.metrics?._score === 'number')
      .sort((a, b) => (b.metrics?._score ?? -Infinity) - (a.metrics?._score ?? -Infinity))
      .slice(0, 8)

    return {
      counts,
      avgScore,
      bestScore,
      avgSharpe,
      bestSharpe,
      top,
    }
  }, [runAlphas])

  const queryClient = useQueryClient()

  // Start task mutation
  const startTaskMutation = useMutation({
    mutationFn: api.startTask,
    onSuccess: (data) => {
      message.success('任务已启动')
      queryClient.invalidateQueries(['task', id])
      queryClient.invalidateQueries(['taskRuns', id])
      if (data?.run_id) {
        setSelectedRunId(data.run_id)
        queryClient.invalidateQueries(['runTrace', data.run_id])
      }
    },
    onError: (err) => {
        message.error(`启动失败: ${err.message}`)
    }
  })

  // Intervene task mutation
  const interveneMutation = useMutation({
    mutationFn: ({ id, action }) => api.interveneTask(id, action),
    onSuccess: (_, variables) => {
      const actionMap = { PAUSE: '暂停', RESUME: '恢复', STOP: '停止' }
      message.success(`任务已${actionMap[variables.action]}`)
      queryClient.invalidateQueries(['task', id])
    },
  })

  const { Panel } = Collapse
  const [activeIterations, setActiveIterations] = React.useState([])
  const lastMaxIterationRef = React.useRef(0)
  
  // Sort and group steps by iteration (consolidate all steps of same iteration)
  const groupedSteps = React.useMemo(() => {
    const traceSteps = runTrace || task?.trace_steps
    if (!traceSteps) return {}
    
    // Sort steps by created_at or id first
    const sortedSteps = [...traceSteps].sort((a, b) => {
      return (a.id || 0) - (b.id || 0)
    })
    
    // First pass: group by iteration only
    const iterGroups = {}
    sortedSteps.forEach(step => {
      const iter = step.iteration || 1
      if (!iterGroups[iter]) {
        iterGroups[iter] = {
          steps: [],
          iteration: iter,
          dataset_ids: new Set(),
          firstCreatedAt: step.created_at
        }
      }
      iterGroups[iter].steps.push(step)
      // Collect all dataset_ids from this iteration
      const datasetId = step.input_data?.dataset_id
      if (datasetId) {
        iterGroups[iter].dataset_ids.add(datasetId)
      }
    })
    
    // Convert to final format with string key
    const groups = {}
    Object.entries(iterGroups).forEach(([iter, group]) => {
      // Use dataset_ids as array, or null if none
      const datasetIds = Array.from(group.dataset_ids)
      const displayDatasetId = datasetIds.length > 0 ? datasetIds.join(', ') : null
      
      groups[iter] = {
        steps: group.steps.sort((a, b) => (a.step_order || 0) - (b.step_order || 0)),
        iteration: parseInt(iter),
        dataset_id: displayDatasetId,
        firstCreatedAt: group.firstCreatedAt
      }
    })
    
    return groups
  }, [runTrace, task?.trace_steps])

  // Get group keys sorted by firstCreatedAt descending (latest first)
  const iterationKeys = React.useMemo(() => {
    return Object.entries(groupedSteps)
      .sort((a, b) => {
        // Sort by firstCreatedAt descending
        const timeA = new Date(a[1].firstCreatedAt || 0).getTime()
        const timeB = new Date(b[1].firstCreatedAt || 0).getTime()
        return timeB - timeA
      })
      .map(([key]) => key)
  }, [groupedSteps])
  
  // Active keys for collapse
  React.useEffect(() => {
    setActiveIterations([])
    lastMaxIterationRef.current = 0
  }, [selectedRunId])

  React.useEffect(() => {
    if (iterationKeys.length > 0) {
      const latestKey = iterationKeys[0]
      // Only auto-expand if we see a NEW group (or first load)
      if (latestKey !== lastMaxIterationRef.current) {
        setActiveIterations([latestKey])
        lastMaxIterationRef.current = latestKey
      }
    }
  }, [iterationKeys])

  if (isLoading) {
    return (
      <div style={{ textAlign: 'center', padding: 100 }}>
        <Spin size="large" />
      </div>
    )
  }

  if (error || !task) {
    return (
      <Empty description="任务未找到">
        <Button onClick={() => navigate('/tasks')}>返回任务列表</Button>
      </Empty>
    )
  }

  const getStatusIcon = (status) => {
    switch (status) {
      case 'SUCCESS':
        return <CheckCircleOutlined style={{ color: '#00ff88' }} />
      case 'FAILED':
        return <CloseCircleOutlined style={{ color: '#ff4757' }} />
      case 'RUNNING':
        return <LoadingOutlined style={{ color: '#00d4ff' }} spin />
      default:
        return null
    }
  }



  return (
    <div>
      {/* Header */}
      <Row justify="space-between" align="middle" style={{ marginBottom: 24 }}>
        <Col>
          <Space>
            <Button icon={<ArrowLeftOutlined />} onClick={() => navigate('/tasks')}>
              返回
            </Button>
            <Title level={3} style={{ margin: 0 }}>
              {task.task_name}
            </Title>
            <Tag color={statusColors[task.status] || 'default'}>{task.status}</Tag>
          </Space>
        </Col>
        <Col>
          <Space>
            {task.status === 'PENDING' && (
              <Button 
                type="primary" 
                icon={<PlayCircleOutlined />}
                loading={startTaskMutation.isLoading}
                onClick={() => startTaskMutation.mutate(task.id)}
              >
                启动
              </Button>
            )}
            {task.status === 'RUNNING' && (
              <Button 
                icon={<PauseCircleOutlined />}
                loading={interveneMutation.isLoading}
                onClick={() => interveneMutation.mutate({ id: task.id, action: 'PAUSE' })}
              >
                暂停
              </Button>
            )}
            {task.status === 'PAUSED' && (
              <Button 
                type="primary"
                icon={<PlayCircleOutlined />}
                loading={interveneMutation.isLoading}
                onClick={() => interveneMutation.mutate({ id: task.id, action: 'RESUME' })}
              >
                恢复
              </Button>
            )}
            {['RUNNING', 'PAUSED'].includes(task.status) && (
              <Button 
                danger 
                icon={<StopOutlined />}
                loading={interveneMutation.isLoading}
                onClick={() => interveneMutation.mutate({ id: task.id, action: 'STOP' })}
              >
                停止
              </Button>
            )}
            {['FAILED', 'COMPLETED', 'STOPPED'].includes(task.status) && (
              <Button 
                type="primary" 
                icon={<ReloadOutlined />} 
                loading={startTaskMutation.isLoading}
                onClick={() => startTaskMutation.mutate(task.id)}
              >
                {task.status === 'COMPLETED' ? '重新运行' : '重试'}
              </Button>
            )}
          </Space>
        </Col>
      </Row>

      <Row gutter={[16, 16]}>
        {/* Left: Task Info */}
        <Col xs={24} lg={8}>
          <Card className="glass-card" title="任务详情">
            <Descriptions column={1} size="small">
              <Descriptions.Item label="地区">{task.region}</Descriptions.Item>
              <Descriptions.Item label="股票池">{task.universe}</Descriptions.Item>
              <Descriptions.Item label="策略">
                <Tag color={task.dataset_strategy === 'AUTO' ? 'cyan' : 'purple'}>
                  {task.dataset_strategy}
                </Tag>
              </Descriptions.Item>
              <Descriptions.Item label="模式">
                {task.agent_mode}
              </Descriptions.Item>
              <Descriptions.Item label="进度">
                <Text strong style={{ color: '#00d4ff' }}>
                  {task.progress_current} / {task.daily_goal}
                </Text>
              </Descriptions.Item>
              <Descriptions.Item label="已发现 Alpha">
                {task.alphas_count || 0}
              </Descriptions.Item>
              <Descriptions.Item label="最大迭代">
                {task.max_iterations || 1}
              </Descriptions.Item>
              <Descriptions.Item label="当前迭代">
                {task.current_iteration || 0} / {task.max_iterations || 1}
              </Descriptions.Item>
              <Descriptions.Item label="创建时间">
                {new Date(task.created_at).toLocaleString()}
              </Descriptions.Item>
              <Descriptions.Item label="更新时间">
                {task.updated_at ? new Date(task.updated_at).toLocaleString() : '--'}
              </Descriptions.Item>
            </Descriptions>
          </Card>

          <Card className="glass-card" title="本次 Run 摘要" style={{ marginTop: 16 }}>
            {!selectedRunId ? (
              <Empty description="未选择 Run" />
            ) : isRunAlphasLoading ? (
              <div style={{ textAlign: 'center', padding: 24 }}>
                <Spin />
              </div>
            ) : (
              <>
                <Descriptions column={1} size="small">
                  <Descriptions.Item label="Run ID">{selectedRunId}</Descriptions.Item>
                  <Descriptions.Item label="Run 状态">
                    <Tag color={selectedRun?.status === 'FAILED' ? 'red' : (selectedRun?.status === 'COMPLETED' ? 'green' : 'blue')}>
                      {selectedRun?.status || '--'}
                    </Tag>
                  </Descriptions.Item>
                  {selectedRun?.error_message && (
                    <Descriptions.Item label="错误原因">
                      <Text type="danger">{selectedRun.error_message}</Text>
                    </Descriptions.Item>
                  )}
                  <Descriptions.Item label="数量">
                    <Space wrap>
                      <Tag color="green">PASS {runAlphaSummary.counts.PASS}</Tag>
                      <Tag color="gold">OPTIMIZE {runAlphaSummary.counts.OPTIMIZE}</Tag>
                      <Tag color="red">FAIL {runAlphaSummary.counts.FAIL}</Tag>
                      {runAlphaSummary.counts.OTHER > 0 && (
                        <Tag>OTHER {runAlphaSummary.counts.OTHER}</Tag>
                      )}
                    </Space>
                  </Descriptions.Item>
                  <Descriptions.Item label="Score">
                    <Space wrap>
                      <Tag>Avg {runAlphaSummary.avgScore?.toFixed?.(3) ?? '--'}</Tag>
                      <Tag>Best {runAlphaSummary.bestScore?.toFixed?.(3) ?? '--'}</Tag>
                    </Space>
                  </Descriptions.Item>
                  <Descriptions.Item label="Sharpe">
                    <Space wrap>
                      <Tag>Avg {runAlphaSummary.avgSharpe?.toFixed?.(2) ?? '--'}</Tag>
                      <Tag>Best {runAlphaSummary.bestSharpe?.toFixed?.(2) ?? '--'}</Tag>
                    </Space>
                  </Descriptions.Item>
                </Descriptions>

                <div style={{ marginTop: 12 }}>
                  <Table
                    size="small"
                    pagination={false}
                    dataSource={runAlphaSummary.top.map(a => ({
                      key: a.id,
                      id: a.id,
                      alpha_id: a.alpha_id,
                      quality_status: a.quality_status,
                      score: a.metrics?._score,
                      sharpe: a.metrics?.sharpe,
                      turnover: a.metrics?.turnover,
                    }))}
                    columns={[
                      {
                        title: 'ID',
                        dataIndex: 'alpha_id',
                        key: 'alpha_id',
                        width: 90,
                        render: (v, r) => v || `#${r.id}`,
                      },
                      {
                        title: '状态',
                        dataIndex: 'quality_status',
                        key: 'quality_status',
                        width: 90,
                        render: (s) => (
                          <Tag color={s === 'PASS' ? 'green' : (s === 'OPTIMIZE' ? 'gold' : 'red')}>{s}</Tag>
                        ),
                      },
                      {
                        title: 'Score',
                        dataIndex: 'score',
                        key: 'score',
                        width: 80,
                        render: (v) => (typeof v === 'number' ? v.toFixed(3) : '--'),
                      },
                      {
                        title: 'Sharpe',
                        dataIndex: 'sharpe',
                        key: 'sharpe',
                        width: 80,
                        render: (v) => (typeof v === 'number' ? v.toFixed(2) : '--'),
                      },
                      {
                        title: 'TO',
                        dataIndex: 'turnover',
                        key: 'turnover',
                        width: 70,
                        render: (v) => (typeof v === 'number' ? v.toFixed(2) : '--'),
                      },
                    ]}
                  />
                </div>
              </>
            )}
          </Card>
        </Col>

        {/* Right: Trace Timeline */}
        <Col xs={24} lg={16}>
          <Card 
            className="glass-card" 
            title="挖掘轨迹 (进化循环)"
            extra={(
              <Space>
                <Text type="secondary">Run</Text>
                <Select
                  size="small"
                  style={{ width: 260 }}
                  value={selectedRunId}
                  onChange={setSelectedRunId}
                  options={(runs || []).map(r => ({
                    value: r.id,
                    label: `#${r.id} ${r.status} ${r.started_at ? new Date(r.started_at).toLocaleString() : ''}`,
                  }))}
                />
                <Text type="secondary">共 {(runTrace || task.trace_steps)?.length || 0} 步 / {iterationKeys.length} 轮</Text>
              </Space>
            )}
          >
            {(runTrace || task.trace_steps) && (runTrace || task.trace_steps).length > 0 ? (
              <Collapse 
                bordered={false} 
                activeKey={activeIterations} 
                onChange={setActiveIterations}
                className="site-collapse-custom-collapse"
                style={{ background: 'transparent' }}
              >
                {iterationKeys.map(groupKey => {
                   const group = groupedSteps[groupKey]
                   const steps = group.steps
                   const iter = group.iteration
                   const datasetId = group.dataset_id
                   
                   // Try to find summary step for this iteration
                   const summaryStep = steps.find(s => s.step_type === 'ROUND_SUMMARY')
                   
                   const header = (
                     <Space>
                       <Text strong>第 {iter} 轮</Text>
                       {datasetId && (
                         <Tag color="purple" style={{ fontSize: 11 }}>{datasetId}</Tag>
                       )}
                       {summaryStep && (summaryStep.output_data?.round_metrics?.success_rate ?? summaryStep.output_data?.success_rate) !== undefined && (
                         <Tag color={(summaryStep.output_data?.round_metrics?.success_rate ?? summaryStep.output_data?.success_rate) > 0 ? 'green' : 'orange'}>
                           成功率: {((summaryStep.output_data?.round_metrics?.success_rate ?? summaryStep.output_data?.success_rate) * 100).toFixed(0)}%
                         </Tag>
                       )}
                       <Text type="secondary" style={{ fontSize: 12 }}>({steps.length} 步)</Text>
                     </Space>
                   )

                   return (
                     <Panel header={header} key={groupKey} style={{ marginBottom: 12, border: '1px solid #303030', borderRadius: 8 }}>
                       <Timeline mode="left" style={{ marginTop: 16 }}>
                         {steps.map((step) => (
                           <Timeline.Item
                             key={step.id}
                             dot={getStatusIcon(step.status)}
                             color={statusColors[step.status] || 'gray'}
                           >
                             <Card 
                               size="small" 
                               style={{ 
                                 background: step.step_type === 'ROUND_SUMMARY' ? 'rgba(0, 50, 20, 0.2)' : 'rgba(0,0,0,0.2)',
                                 marginBottom: 8,
                                 borderColor: step.step_type === 'ROUND_SUMMARY' ? '#004d26' : undefined
                               }}
                             >
                               <Space>
                                 {stepIcons[step.step_type]}
                                 <Text strong>
                                    {step.step_type === 'ROUND_SUMMARY' ? '本轮总结' : `Step ${step.step_order}: ${step.step_type}`}
                                 </Text>
                                 <Tag>{step.duration_ms ? `${step.duration_ms}ms` : '--'}</Tag>
                               </Space>
                               
                               {/* Rich Content Rendering */}
                               
                               {/* RAG_QUERY: Show top patterns and pitfalls */}
                               {step.step_type === 'RAG_QUERY' && (
                                 <div style={{ marginTop: 8 }}>
                                   {step.output_data?.top_patterns?.length > 0 ? (
                                     <>
                                       <Text type="secondary" style={{ fontSize: 12 }}>参考模式:</Text>
                                       <ul style={{ paddingLeft: 20, margin: '4px 0', fontSize: 12, color: 'rgba(255,255,255,0.6)' }}>
                                         {step.output_data.top_patterns.map((p, i) => (
                                           <li key={i}>{p}</li>
                                         ))}
                                       </ul>
                                     </>
                                   ) : (
                                      <Text type="secondary" style={{ fontSize: 12, marginRight: 8 }}>暂无参考模式</Text>
                                   )}
                                   
                                   {step.output_data?.top_pitfalls?.length > 0 && (
                                     <>
                                       <Text type="secondary" style={{ fontSize: 12, marginTop: 8, display: 'block' }}>避坑指南:</Text>
                                       <ul style={{ paddingLeft: 20, margin: '4px 0', fontSize: 12, color: '#ff7875' }}>
                                         {step.output_data.top_pitfalls.map((p, i) => (
                                           <li key={i}>{p}</li>
                                         ))}
                                       </ul>
                                     </>
                                   )}
                                 </div>
                               )}
         
                               {/* DISTILL_CONTEXT: Show reasoning and selected concepts */}
                               {step.step_type === 'DISTILL_CONTEXT' && (
                                 <div style={{ marginTop: 8 }}>
                                   {step.output_data?.reasoning && (
                                     <Paragraph 
                                       ellipsis={{ rows: 2, expandable: true, symbol: '展开' }} 
                                       style={{ fontSize: 13, color: 'rgba(255,255,255,0.85)', fontStyle: 'italic', marginBottom: 8 }}
                                     >
                                        "{step.output_data.reasoning}"
                                     </Paragraph>
                                   )}
                                   {step.output_data?.selected_concepts && (
                                     <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
                                       {step.output_data.selected_concepts.map((c, i) => (
                                         <Tag key={i} color="blue" style={{ fontSize: 11 }}>{c}</Tag>
                                       ))}
                                     </div>
                                   )}
                                 </div>
                               )}
         
                               {/* HYPOTHESIS: Show generated hypotheses */}
                               {step.step_type === 'HYPOTHESIS' && (
                                 <div style={{ marginTop: 8 }}>
                                    {step.output_data?.hypotheses?.map((h, i) => {
                                      const content = typeof h === 'string' ? h : (h.statement || h.idea || JSON.stringify(h));
                                      const rationale = typeof h === 'object' && h.rationale ? h.rationale : null;
                                      
                                      return (
                                        <div key={i} style={{ marginBottom: 4 }}>
                                          <Paragraph ellipsis={{ rows: 2, expandable: true, symbol: '展开' }} style={{ fontSize: 13, marginBottom: 0 }}>
                                            <Text strong style={{ color: '#00d4ff', marginRight: 8 }}>H{i+1}:</Text>
                                            {content}
                                          </Paragraph>
                                          {rationale && (
                                            <Text type="secondary" style={{ fontSize: 12, marginLeft: 22 }}>
                                              {rationale}
                                            </Text>
                                          )}
                                        </div>
                                      );
                                    })}
                                    {/* Legacy support */}
                                    {!step.output_data?.hypotheses && step.output_data?.hypothesis && (
                                       <Paragraph ellipsis={{ rows: 2 }} style={{ fontSize: 13 }}>
                                         💡 {step.output_data.hypothesis}
                                       </Paragraph>
                                    )}
                                 </div>
                               )}
                               
                               {/* CODE_GEN: Show expressions */}
                               {step.step_type === 'CODE_GEN' && step.output_data?.expressions && (
                                 <div style={{ marginTop: 8 }}>
                                    {step.output_data.expressions.map((expr, i) => (
                                      <pre key={i} style={{ 
                                        fontSize: 11, 
                                        background: '#1f1f1f', 
                                        padding: 4, 
                                        borderRadius: 4,
                                        marginBottom: 4,
                                        overflowX: 'auto'
                                      }}>
                                        {expr}
                                      </pre>
                                    ))}
                                 </div>
                               )}
                               
                               {/* SIMULATE: Show Results with Metrics */}
                               {step.step_type === 'SIMULATE' && step.output_data?.results && (
                                 <div style={{ marginTop: 8 }}>
                                   <Text type="secondary" style={{ fontSize: 12 }}>
                                     模拟结果: {step.output_data.success_count || 0} 成功
                                   </Text>
                                   {step.output_data.results.map((r, i) => (
                                     <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 8, marginTop: 4 }}>
                                       <Tag color={r.err ? 'red' : 'blue'} style={{ fontSize: 11 }}>
                                         {r.id || `#${i+1}`}
                                       </Tag>
                                       {r.metrics && (
                                         <Space size="small" wrap>
                                           <Tag color={r.metrics.sharpe >= 1.2 ? 'green' : (r.metrics.sharpe >= 0 ? 'orange' : 'red')}>
                                             Sharpe: {r.metrics.sharpe?.toFixed(2) ?? '--'}
                                           </Tag>
                                           <Tag>Returns: {(r.metrics.returns * 100)?.toFixed(1) ?? '--'}%</Tag>
                                           <Tag>Turnover: {r.metrics.turnover?.toFixed(2) ?? '--'}</Tag>
                                           <Tag>Fitness: {r.metrics.fitness?.toFixed(2) ?? '--'}</Tag>
                                         </Space>
                                       )}
                                       {r.err && <Text type="danger" style={{ fontSize: 11 }}>{r.err}</Text>}
                                     </div>
                                   ))}
                                 </div>
                               )}

                               {step.step_type === 'EVALUATE' && step.output_data?.details && (
                                 <div style={{ marginTop: 8 }}>
                                   <Text type="secondary" style={{ fontSize: 12 }}>
                                     评估结果: ✅ {step.output_data.pass_count || 0} 通过, ⚡ {step.output_data.optimize_count || 0} 优化, ❌ {step.output_data.fail_count || 0} 失败
                                   </Text>
                                   {step.output_data.details.map((d, i) => (
                                     <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 8, marginTop: 4, flexWrap: 'wrap' }}>
                                       <Tag
                                         color={d.status === 'PASS' ? 'green' : (d.status === 'OPTIMIZE' ? 'gold' : 'red')}
                                         style={{ fontSize: 11 }}
                                       >
                                         {d.status} {d.id || `#${i+1}`}
                                       </Tag>
                                       <Space size="small" wrap>
                                         <Tag>Score: {d.score?.toFixed?.(3) ?? d.score ?? '--'}</Tag>
                                         <Tag color={d.sharpe >= 1.5 ? 'green' : 'default'}>Sharpe: {d.sharpe?.toFixed(2) ?? '--'}</Tag>
                                         <Tag color={d.turnover <= 0.3 ? 'green' : 'orange'}>Turnover: {d.turnover?.toFixed(2) ?? '--'}</Tag>
                                         <Tag>Fitness: {d.fitness?.toFixed(2) ?? '--'}</Tag>
                                       </Space>
                                       {d.optimize_reason && (
                                         <Text type="secondary" style={{ fontSize: 11 }}>
                                           {d.optimize_reason}
                                         </Text>
                                       )}
                                     </div>
                                   ))}
                                 </div>
                               )}

                               {/* ROUND_SUMMARY: Show Rich Round Stats & Intelligent Strategy */}
                               {step.step_type === 'ROUND_SUMMARY' && step.output_data && (
                                 <div style={{ marginTop: 12 }}>
                                   <Row gutter={[12, 12]}>
                                     {/* Left: Performance Metrics */}
                                     <Col span={12}>
                                       <div style={{ background: 'rgba(0,0,0,0.2)', padding: 10, borderRadius: 4 }}>
                                         <Text type="secondary" style={{ fontSize: 12, fontWeight: 'bold' }}>本轮战绩</Text>
                                         <div style={{ marginTop: 6 }}>
                                            <Tag color={(step.output_data.round_metrics?.success_rate ?? step.output_data.success_rate ?? 0) > 0 ? "green" : "red"} style={{ marginRight: 4 }}>
                                              {(step.output_data.round_metrics?.passed_count ?? step.output_data.succeeded_alphas ?? 0) > 0 ? "MINING SUCCESS" : "MINING FAIL"}
                                            </Tag>
                                            <Text style={{ fontSize: 12 }}>
                                              Alphas: {step.output_data.round_metrics?.total_simulated ?? step.output_data.simulated_alphas ?? 0} (通过 {step.output_data.round_metrics?.passed_count ?? step.output_data.succeeded_alphas ?? 0})
                                            </Text>
                                         </div>
                                         
                                         {/* Multi-dimensional Quality Metrics */}
                                         <div style={{ marginTop: 8, display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 4 }}>
                                            <Text style={{ fontSize: 11 }}>
                                              Best Sharpe: <span style={{ color: '#00ff88', fontWeight: 'bold' }}>{(step.output_data.round_metrics?.best_sharpe ?? step.output_data.best_sharpe)?.toFixed?.(2) ?? 'N/A'}</span>
                                            </Text>
                                            <Text style={{ fontSize: 11 }}>
                                              Avg Sharpe: <span style={{ color: '#87d068' }}>{(step.output_data.round_metrics?.avg_sharpe ?? step.output_data.avg_sharpe)?.toFixed?.(2) ?? 'N/A'}</span>
                                            </Text>
                                            <Text style={{ fontSize: 11 }}>
                                              Best Fitness: <span style={{ color: '#00d4ff' }}>{(step.output_data.round_metrics?.best_fitness ?? step.output_data.best_fitness)?.toFixed?.(2) ?? 'N/A'}</span>
                                            </Text>
                                            <Text style={{ fontSize: 11 }}>
                                              Avg Fitness: <span style={{ color: '#69c0ff' }}>{(step.output_data.round_metrics?.avg_fitness ?? step.output_data.avg_fitness)?.toFixed?.(2) ?? 'N/A'}</span>
                                            </Text>
                                            <Text style={{ fontSize: 11 }}>
                                              Avg Turnover: <span style={{ color: '#faad14' }}>{(step.output_data.round_metrics?.avg_turnover ?? step.output_data.avg_turnover)?.toFixed?.(2) ?? 'N/A'}</span>
                                            </Text>
                                            <Text style={{ fontSize: 11 }}>
                                              Avg Returns: <span style={{ color: '#b37feb' }}>{(step.output_data.round_metrics?.avg_returns ?? step.output_data.avg_returns) ? ((step.output_data.round_metrics?.avg_returns ?? step.output_data.avg_returns) * 100).toFixed(1) + '%' : 'N/A'}</span>
                                            </Text>
                                         </div>
                                         
                                         {/* Failure Analysis */}
                                         {(step.output_data.error_breakdown || step.output_data.round_metrics) && (
                                           <div style={{ marginTop: 8, borderTop: '1px solid #303030', paddingTop: 6 }}>
                                             <Text type="secondary" style={{ fontSize: 11 }}>错误分析:</Text>
                                             <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap', marginTop: 2 }}>
                                               {(step.output_data.error_breakdown?.syntax_errors ?? step.output_data.round_metrics?.syntax_errors ?? 0) > 0 && (
                                                 <Tag color="red" style={{ fontSize: 10 }}>语法: {step.output_data.error_breakdown?.syntax_errors ?? step.output_data.round_metrics?.syntax_errors}</Tag>
                                               )}
                                               {(step.output_data.error_breakdown?.simulation_errors ?? step.output_data.round_metrics?.simulation_errors ?? 0) > 0 && (
                                                 <Tag color="orange" style={{ fontSize: 10 }}>模拟: {step.output_data.error_breakdown?.simulation_errors ?? step.output_data.round_metrics?.simulation_errors}</Tag>
                                               )}
                                               {(step.output_data.error_breakdown?.quality_failures ?? step.output_data.round_metrics?.quality_failures ?? 0) > 0 && (
                                                 <Tag color="gold" style={{ fontSize: 10 }}>质量: {step.output_data.error_breakdown?.quality_failures ?? step.output_data.round_metrics?.quality_failures}</Tag>
                                               )}
                                             </div>
                                           </div>
                                         )}
                                         
                                         {/* Problematic Fields */}
                                         {step.output_data.problematic_fields?.length > 0 && (
                                           <div style={{ marginTop: 4 }}>
                                             <Text type="secondary" style={{ fontSize: 10 }}>问题字段: </Text>
                                             {step.output_data.problematic_fields.slice(0, 3).map((f, i) => (
                                               <Tag key={i} color="volcano" style={{ fontSize: 9 }}>{f}</Tag>
                                             ))}
                                           </div>
                                         )}
                                       </div>
                                     </Col>
                                     
                                     {/* Right: Intelligent Strategy */}
                                     <Col span={12}>
                                        <div style={{ background: 'rgba(0,0,0,0.2)', padding: 10, borderRadius: 4 }}>
                                          <Text type="secondary" style={{ fontSize: 12, fontWeight: 'bold' }}>下轮策略 (RD-Agent Style)</Text>
                                          {(step.output_data.next_strategy || step.output_data.next_action || step.output_data.next_reasoning) ? (
                                             <div style={{ marginTop: 6, display: 'flex', flexDirection: 'column', gap: 6 }}>
                                                {/* Core Parameters */}
                                                <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                                                  <Tag color="geekblue">Temp: {step.output_data.next_strategy?.temperature?.toFixed(1) ?? 'N/A'}</Tag>
                                                  <Tag color="purple">Exploration: {step.output_data.next_strategy?.exploration_weight?.toFixed(1) ?? 'N/A'}</Tag>
                                                </div>
                                                
                                                {/* Action Summary */}
                                                {(step.output_data.next_strategy?.action || step.output_data.next_action) && (
                                                  <Text style={{ fontSize: 11, color: '#00d4ff' }}>
                                                    {step.output_data.next_strategy?.action || step.output_data.next_action}
                                                  </Text>
                                                )}
                                                
                                                {/* Reasoning */}
                                                {(step.output_data.next_strategy?.reasoning || step.output_data.next_reasoning) && (
                                                  <Paragraph 
                                                    ellipsis={{ rows: 2, expandable: true, symbol: '展开' }}
                                                    style={{ fontSize: 10, color: 'rgba(255,255,255,0.65)', marginBottom: 0 }}
                                                  >
                                                    {step.output_data.next_strategy?.reasoning || step.output_data.next_reasoning}
                                                  </Paragraph>
                                                )}
                                                
                                                {/* Focus Areas */}
                                                {asArray(step.output_data.next_strategy?.focus_hypotheses).length > 0 && (
                                                  <div>
                                                    <Text type="secondary" style={{ fontSize: 10 }}>🎯 聚焦方向:</Text>
                                                    <div style={{ marginTop: 2 }}>
                                                      {asArray(step.output_data.next_strategy?.focus_hypotheses).slice(0, 2).map((h, i) => (
                                                        <Tag key={i} color="cyan" style={{ fontSize: 9, marginBottom: 2 }}>{h}</Tag>
                                                      ))}
                                                    </div>
                                                  </div>
                                                )}
                                                
                                                {/* Amplify & Avoid Patterns */}
                                                <div style={{ display: 'flex', gap: 8 }}>
                                                {asArray(step.output_data.next_strategy?.amplify_patterns).length > 0 && (
                                                    <div style={{ flex: 1 }}>
                                                      <Text type="secondary" style={{ fontSize: 10 }}>✅ 强化:</Text>
                                                      {asArray(step.output_data.next_strategy?.amplify_patterns).slice(0, 2).map((p, i) => (
                                                        <Tag key={i} color="green" style={{ fontSize: 9, display: 'block', marginTop: 2 }}>{p}</Tag>
                                                      ))}
                                                    </div>
                                                  )}
                                                  {asArray(step.output_data.next_strategy?.avoid_patterns).length > 0 && (
                                                    <div style={{ flex: 1 }}>
                                                      <Text type="secondary" style={{ fontSize: 10 }}>❌ 避免:</Text>
                                                      {asArray(step.output_data.next_strategy?.avoid_patterns).slice(0, 2).map((p, i) => (
                                                        <Tag key={i} color="red" style={{ fontSize: 9, display: 'block', marginTop: 2 }}>{p}</Tag>
                                                      ))}
                                                    </div>
                                                  )}
                                                </div>
                                                
                                                {/* Optimization Suggestions */}
                                                {asArray(step.output_data.next_strategy?.optimization_suggestions).length > 0 && (
                                                  <div style={{ borderTop: '1px solid #303030', paddingTop: 4 }}>
                                                    <Text type="secondary" style={{ fontSize: 10 }}>💡 优化建议:</Text>
                                                    {asArray(step.output_data.next_strategy?.optimization_suggestions).slice(0, 1).map((s, i) => (
                                                      <Text key={i} style={{ fontSize: 10, display: 'block', color: '#fadb14' }}>
                                                        [{s.type}] {s.suggestion}
                                                      </Text>
                                                    ))}
                                                  </div>
                                                )}
                                             </div>
                                          ) : (
                                            <div style={{ marginTop: 4 }}>
                                              <Text type="secondary" style={{ fontSize: 12 }}>迭代完成或无新策略</Text>
                                            </div>
                                          )}
                                        </div>
                                     </Col>
                                   </Row>
                                 </div>
                               )}
         
                               {/* Legacy single expression display */}
                               {!['ROUND_SUMMARY', 'CODE_GEN'].includes(step.step_type) && !step.output_data?.expressions && (step.output_data?.expression || step.input_data?.expression) && (
                                 <pre style={{ 
                                   marginTop: 8, 
                                   marginBottom: 0,
                                   fontSize: 12,
                                   maxHeight: 100,
                                   overflow: 'auto',
                                   background: '#1f1f1f',
                                   padding: 5
                                 }}>
                                   {step.output_data?.expression || step.input_data?.expression}
                                 </pre>
                               )}
                               
                               {step.error_message && (
                                 <Text type="danger" style={{ display: 'block', marginTop: 8, fontSize: 12 }}>
                                   ❌ {step.error_message}
                                 </Text>
                               )}
                             </Card>
                           </Timeline.Item>
                         ))}
                       </Timeline>
                   </Panel>
                   )
                })}
              </Collapse>
            ) : (
              <Empty description="暂无轨迹记录" />
            )}
          </Card>
        </Col>
      </Row>
    </div>
  )
}
