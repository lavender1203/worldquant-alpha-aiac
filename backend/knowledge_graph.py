"""
RD-Agent Style Knowledge Graph for Alpha Mining

This module implements a graph-based knowledge management system inspired by RD-Agent's CoSTEER framework.
It provides structured storage and retrieval for:
1. Components (operators, patterns, field combinations)
2. Errors (failure patterns with fixes)
3. Tasks (hypothesis-implementation pairs)
4. Successful implementations (for future reference)

Key features:
- Graph structure with nodes and edges
- Similarity-based retrieval
- Experience accumulation over time
- Multi-dimensional queries (former trace, component, error)

Reference: RD-Agent's knowledge management in rdagent/components/coder/model_coder/
"""

from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
import hashlib
import json
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_

from backend.models import KnowledgeEntry, Alpha


# =============================================================================
# Node Types
# =============================================================================

@dataclass
class KnowledgeNode:
    """Base class for knowledge graph nodes."""
    id: str
    content: str
    node_type: str = ""  # "component" | "error" | "task" | "implementation" - default empty, set by subclasses
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    usage_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    
    @property
    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.5
    
    def fingerprint(self) -> str:
        """Generate unique fingerprint for deduplication."""
        return hashlib.md5(f"{self.node_type}:{self.content}".encode()).hexdigest()[:16]


@dataclass
class ComponentNode(KnowledgeNode):
    """Represents a reusable component (operator pattern, field combination)."""
    component_type: str = ""  # "operator_chain" | "field_combo" | "wrapper_pattern"
    operators: List[str] = field(default_factory=list)
    fields: List[str] = field(default_factory=list)
    template: str = ""  # Generalized template with placeholders
    
    def __post_init__(self):
        self.node_type = "component"


@dataclass
class ErrorNode(KnowledgeNode):
    """Represents a known error pattern with solution."""
    error_type: str = ""  # "syntax" | "semantic" | "quality" | "simulation"
    error_pattern: str = ""  # Regex or substring to match
    solution: str = ""  # Recommended fix
    severity: str = "medium"  # "low" | "medium" | "high"
    
    def __post_init__(self):
        self.node_type = "error"


@dataclass
class TaskNode(KnowledgeNode):
    """Represents a hypothesis-implementation task."""
    hypothesis: str = ""
    implementation: str = ""  # Expression
    result: str = ""  # "success" | "failure" | "partial"
    metrics: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        self.node_type = "task"


@dataclass 
class ImplementationNode(KnowledgeNode):
    """Represents a successful implementation for reference."""
    expression: str = ""
    hypothesis: str = ""
    dataset_category: str = ""
    sharpe: float = 0.0
    fitness: float = 0.0
    key_operators: List[str] = field(default_factory=list)
    key_fields: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        self.node_type = "implementation"


@dataclass
class KnowledgeEdge:
    """Edge connecting two knowledge nodes."""
    source_id: str
    target_id: str
    edge_type: str  # "derived_from" | "similar_to" | "fixes" | "implements"
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Knowledge Graph
# =============================================================================

class AlphaKnowledgeGraph:
    """
    RD-Agent style knowledge graph for alpha mining.
    
    Provides three key query types:
    1. Former Trace Query: Find similar past tasks/implementations
    2. Component Query: Find reusable components for new tasks
    3. Error Query: Find solutions for known error patterns
    
    Usage:
        graph = AlphaKnowledgeGraph(db)
        await graph.initialize()
        
        # Query for similar past implementations
        similar = await graph.query_former_trace(hypothesis, dataset_category)
        
        # Query for useful components
        components = await graph.query_components(field_types, target_signal)
        
        # Query for error solutions
        fix = await graph.query_error_solution(error_message)
    """
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self.nodes: Dict[str, KnowledgeNode] = {}
        self.edges: List[KnowledgeEdge] = []
        self.node_index: Dict[str, Set[str]] = defaultdict(set)  # type -> node_ids
        self._initialized = False
    
    async def initialize(self):
        """Load existing knowledge from database into graph."""
        logger.info("[KnowledgeGraph] Initializing from database...")
        
        # Load knowledge entries
        query = select(KnowledgeEntry).where(KnowledgeEntry.is_active == True)
        result = await self.db.execute(query)
        entries = result.scalars().all()
        
        for entry in entries:
            node = self._entry_to_node(entry)
            if node:
                self._add_node(node)
        
        # Load successful alphas as implementation nodes
        alpha_query = select(Alpha).where(Alpha.quality_status == "PASS").limit(100)
        alpha_result = await self.db.execute(alpha_query)
        alphas = alpha_result.scalars().all()
        
        for alpha in alphas:
            impl_node = ImplementationNode(
                id=f"impl_{alpha.id}",
                content=alpha.expression,
                expression=alpha.expression,
                hypothesis=alpha.hypothesis or "",
                dataset_category=alpha.dataset_id or "",
                sharpe=alpha.is_sharpe or 0,
                fitness=alpha.is_fitness or 0,
                key_operators=self._extract_operators(alpha.expression),
                key_fields=self._extract_fields(alpha.expression),
            )
            self._add_node(impl_node)
        
        self._initialized = True
        logger.info(
            f"[KnowledgeGraph] Initialized | "
            f"nodes={len(self.nodes)} edges={len(self.edges)}"
        )
    
    def _entry_to_node(self, entry: KnowledgeEntry) -> Optional[KnowledgeNode]:
        """Convert KnowledgeEntry to appropriate node type."""
        metadata = entry.meta_data or {}
        
        if entry.entry_type == "SUCCESS_PATTERN":
            return ComponentNode(
                id=f"comp_{entry.id}",
                content=entry.pattern or "",
                component_type="operator_chain",
                template=metadata.get("template", entry.pattern),
                operators=self._extract_operators(entry.pattern or ""),
                usage_count=entry.usage_count or 0,
                success_count=metadata.get("success_count", 1),
                metadata=metadata,
            )
        
        elif entry.entry_type == "FAILURE_PITFALL":
            return ErrorNode(
                id=f"err_{entry.id}",
                content=entry.description or "",
                error_type=metadata.get("error_type", "unknown"),
                error_pattern=entry.pattern or "",
                solution=metadata.get("recommendation", ""),
                severity=metadata.get("severity", "medium"),
                failure_count=metadata.get("failure_count", 1),
                metadata=metadata,
            )
        
        return None
    
    def _add_node(self, node: KnowledgeNode):
        """Add node to graph with indexing."""
        self.nodes[node.id] = node
        self.node_index[node.node_type].add(node.id)
    
    def _add_edge(self, edge: KnowledgeEdge):
        """Add edge to graph."""
        self.edges.append(edge)
    
    def _extract_operators(self, expression: str) -> List[str]:
        """Extract operator names from expression."""
        import re
        pattern = r'\b(ts_\w+|group_\w+|vec_\w+|rank|zscore|log|abs|sign)\b'
        return list(set(re.findall(pattern, expression.lower())))
    
    def _extract_fields(self, expression: str) -> List[str]:
        """Extract potential field names from expression."""
        import re
        # Look for identifiers that aren't operators
        operators = {'ts_rank', 'ts_delta', 'ts_zscore', 'ts_mean', 'ts_decay_linear', 
                    'vec_sum', 'vec_avg', 'group_neutralize', 'rank', 'divide', 'add'}
        pattern = r'\b([a-z][a-z0-9_]*)\b'
        matches = re.findall(pattern, expression.lower())
        return [m for m in matches if m not in operators and len(m) > 2]
    
    # =========================================================================
    # Query Methods (RD-Agent Style)
    # =========================================================================
    
    async def query_former_trace(
        self,
        hypothesis: str,
        dataset_category: str = None,
        limit: int = 5
    ) -> List[Dict]:
        """
        Query for similar past tasks/implementations.
        
        This is the "Former Trace Query" from RD-Agent:
        - Find past experiments with similar hypotheses
        - Retrieve their results (success/failure)
        - Learn what worked and what didn't
        
        Args:
            hypothesis: Current hypothesis to find similar traces for
            dataset_category: Optional category filter
            limit: Maximum results
        
        Returns:
            List of similar past tasks with their outcomes
        """
        if not self._initialized:
            await self.initialize()
        
        results = []
        hypothesis_lower = hypothesis.lower()
        hypothesis_words = set(hypothesis_lower.split())
        
        # Search implementation and task nodes
        for node_id in self.node_index.get("implementation", set()):
            node = self.nodes[node_id]
            if not isinstance(node, ImplementationNode):
                continue
            
            # Calculate similarity
            similarity = self._calculate_text_similarity(
                hypothesis_lower,
                (node.hypothesis or "").lower()
            )
            
            # Category bonus
            if dataset_category and dataset_category.lower() in (node.dataset_category or "").lower():
                similarity += 0.2
            
            if similarity > 0.1:  # Threshold
                results.append({
                    "type": "former_trace",
                    "hypothesis": node.hypothesis,
                    "expression": node.expression,
                    "result": "success",
                    "sharpe": node.sharpe,
                    "fitness": node.fitness,
                    "similarity": similarity,
                    "key_operators": node.key_operators,
                    "lesson": f"This approach achieved Sharpe={node.sharpe:.2f}"
                })
        
        # Also check task nodes for failures
        for node_id in self.node_index.get("task", set()):
            node = self.nodes[node_id]
            if not isinstance(node, TaskNode):
                continue
            
            similarity = self._calculate_text_similarity(
                hypothesis_lower,
                (node.hypothesis or "").lower()
            )
            
            if similarity > 0.1 and node.result == "failure":
                results.append({
                    "type": "former_trace",
                    "hypothesis": node.hypothesis,
                    "expression": node.implementation,
                    "result": "failure",
                    "similarity": similarity,
                    "lesson": f"This approach failed: {node.content[:100]}"
                })
        
        # Sort by similarity and limit
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:limit]
    
    async def query_components(
        self,
        field_types: List[str] = None,
        target_signal: str = None,
        limit: int = 5
    ) -> List[Dict]:
        """
        Query for reusable components.
        
        This is the "Component Query" from RD-Agent:
        - Find proven operator patterns and field combinations
        - Match by field type and signal type
        - Return templates for use in new expressions
        
        Args:
            field_types: Types of fields available (sentiment, count, etc.)
            target_signal: Target signal type (momentum, mean_reversion)
            limit: Maximum results
        
        Returns:
            List of reusable components with templates
        """
        if not self._initialized:
            await self.initialize()
        
        results = []
        field_types = field_types or []
        
        for node_id in self.node_index.get("component", set()):
            node = self.nodes[node_id]
            if not isinstance(node, ComponentNode):
                continue
            
            # Calculate relevance score
            score = node.success_rate * 0.5
            
            # Field type match
            if field_types:
                node_fields = [f.lower() for f in node.fields]
                for ft in field_types:
                    if any(ft.lower() in nf for nf in node_fields):
                        score += 0.2
            
            # Signal type match
            if target_signal:
                signal_ops = {
                    "momentum": ["ts_delta", "ts_returns", "ts_rank"],
                    "mean_reversion": ["ts_zscore", "ts_mean", "ts_std_dev"],
                    "value": ["rank", "zscore", "divide"],
                }
                target_ops = signal_ops.get(target_signal.lower(), [])
                if any(op in node.operators for op in target_ops):
                    score += 0.3
            
            # Usage bonus
            score += min(node.usage_count / 20, 0.2)
            
            results.append({
                "type": "component",
                "template": node.template or node.content,
                "operators": node.operators,
                "component_type": node.component_type,
                "success_rate": node.success_rate,
                "usage_count": node.usage_count,
                "relevance_score": score,
                "description": node.metadata.get("description", "")
            })
        
        results.sort(key=lambda x: x["relevance_score"], reverse=True)
        return results[:limit]
    
    async def query_error_solution(
        self,
        error_message: str,
        error_type: str = None,
        limit: int = 3
    ) -> List[Dict]:
        """
        Query for solutions to known errors.
        
        This is the "Error Query" from RD-Agent:
        - Match error against known error patterns
        - Return recommended solutions
        - Prioritize by severity and match quality
        
        Args:
            error_message: Error message to find solution for
            error_type: Optional error type filter
            limit: Maximum results
        
        Returns:
            List of matching errors with solutions
        """
        if not self._initialized:
            await self.initialize()
        
        results = []
        error_lower = error_message.lower()
        
        for node_id in self.node_index.get("error", set()):
            node = self.nodes[node_id]
            if not isinstance(node, ErrorNode):
                continue
            
            # Type filter
            if error_type and node.error_type != error_type:
                continue
            
            # Pattern matching
            match_score = 0
            if node.error_pattern:
                if node.error_pattern.lower() in error_lower:
                    match_score = 1.0
                elif any(word in error_lower for word in node.error_pattern.lower().split()):
                    match_score = 0.5
            
            # Content similarity
            content_sim = self._calculate_text_similarity(error_lower, node.content.lower())
            match_score = max(match_score, content_sim)
            
            if match_score > 0.1:
                # Severity bonus
                severity_scores = {"high": 1.0, "medium": 0.7, "low": 0.4}
                match_score += severity_scores.get(node.severity, 0.5) * 0.2
                
                results.append({
                    "type": "error_solution",
                    "error_type": node.error_type,
                    "error_pattern": node.error_pattern,
                    "solution": node.solution,
                    "severity": node.severity,
                    "match_score": match_score,
                    "description": node.content
                })
        
        results.sort(key=lambda x: x["match_score"], reverse=True)
        return results[:limit]
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Simple word overlap similarity."""
        if not text1 or not text2:
            return 0.0
        
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union) if union else 0.0
    
    # =========================================================================
    # Update Methods
    # =========================================================================
    
    async def record_task_result(
        self,
        hypothesis: str,
        expression: str,
        result: str,  # "success" | "failure"
        metrics: Dict[str, float] = None,
        error_info: Dict = None
    ):
        """
        Record task result in knowledge graph.
        
        This enables learning from experiments:
        - Successful tasks become implementation nodes
        - Failed tasks become task nodes for future reference
        - Errors get linked to error nodes
        """
        metrics = metrics or {}
        
        if result == "success":
            # Add implementation node
            impl_node = ImplementationNode(
                id=f"impl_{datetime.now().timestamp()}",
                content=expression,
                expression=expression,
                hypothesis=hypothesis,
                sharpe=metrics.get("sharpe", 0),
                fitness=metrics.get("fitness", 0),
                key_operators=self._extract_operators(expression),
                key_fields=self._extract_fields(expression),
            )
            self._add_node(impl_node)
            
        else:
            # Add task node
            task_node = TaskNode(
                id=f"task_{datetime.now().timestamp()}",
                content=error_info.get("message", "") if error_info else "Failed",
                hypothesis=hypothesis,
                implementation=expression,
                result="failure",
                metrics=metrics,
            )
            self._add_node(task_node)
            
            # Link to error node if applicable
            if error_info:
                solutions = await self.query_error_solution(
                    error_info.get("message", ""),
                    error_info.get("type")
                )
                if solutions:
                    self._add_edge(KnowledgeEdge(
                        source_id=task_node.id,
                        target_id=solutions[0].get("id", ""),
                        edge_type="has_error",
                    ))
        
        logger.debug(f"[KnowledgeGraph] Recorded task result: {result}")
    
    async def add_error_pattern(
        self,
        error_pattern: str,
        error_type: str,
        solution: str,
        severity: str = "medium"
    ):
        """Add new error pattern with solution."""
        error_node = ErrorNode(
            id=f"err_{datetime.now().timestamp()}",
            content=error_pattern,
            error_type=error_type,
            error_pattern=error_pattern,
            solution=solution,
            severity=severity,
        )
        self._add_node(error_node)
        
        # Also persist to database
        entry = KnowledgeEntry(
            entry_type="FAILURE_PITFALL",
            pattern=error_pattern,
            description=solution,
            meta_data={
                "error_type": error_type,
                "severity": severity,
                "recommendation": solution,
                "source": "knowledge_graph",
            }
        )
        self.db.add(entry)
        await self.db.commit()
        
        logger.info(f"[KnowledgeGraph] Added error pattern: {error_pattern[:50]}")
    
    async def add_component(
        self,
        template: str,
        component_type: str,
        description: str = "",
        operators: List[str] = None
    ):
        """Add new reusable component."""
        comp_node = ComponentNode(
            id=f"comp_{datetime.now().timestamp()}",
            content=template,
            component_type=component_type,
            template=template,
            operators=operators or self._extract_operators(template),
        )
        self._add_node(comp_node)
        
        # Also persist to database
        entry = KnowledgeEntry(
            entry_type="SUCCESS_PATTERN",
            pattern=template,
            description=description,
            meta_data={
                "component_type": component_type,
                "operators": operators or self._extract_operators(template),
                "source": "knowledge_graph",
            }
        )
        self.db.add(entry)
        await self.db.commit()
        
        logger.info(f"[KnowledgeGraph] Added component: {template[:50]}")
    
    # =========================================================================
    # Statistics and Export
    # =========================================================================
    
    def get_stats(self) -> Dict[str, Any]:
        """Get knowledge graph statistics."""
        return {
            "total_nodes": len(self.nodes),
            "components": len(self.node_index.get("component", set())),
            "errors": len(self.node_index.get("error", set())),
            "tasks": len(self.node_index.get("task", set())),
            "implementations": len(self.node_index.get("implementation", set())),
            "edges": len(self.edges),
        }
    
    def export_graph(self) -> Dict[str, Any]:
        """Export graph for visualization."""
        return {
            "nodes": [
                {
                    "id": node.id,
                    "type": node.node_type,
                    "content": node.content[:100],
                    "success_rate": node.success_rate,
                }
                for node in self.nodes.values()
            ],
            "edges": [
                {
                    "source": edge.source_id,
                    "target": edge.target_id,
                    "type": edge.edge_type,
                }
                for edge in self.edges
            ]
        }


# =============================================================================
# Factory Function
# =============================================================================

async def create_knowledge_graph(db: AsyncSession) -> AlphaKnowledgeGraph:
    """Factory function to create and initialize knowledge graph."""
    graph = AlphaKnowledgeGraph(db)
    await graph.initialize()
    return graph
