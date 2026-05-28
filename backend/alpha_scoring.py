"""
Alpha Scoring Module - Enhanced with Adaptive Thresholds

Features:
1. Multi-objective scoring for alpha evaluation
2. Adaptive thresholds by region and dataset category
3. Dynamic quality gates based on historical performance
4. Comprehensive pass/fail criteria

Reference: 优化.md Section 3.1, BRAIN Alpha submission requirements
"""

from typing import Dict, Optional, Any, Tuple, List
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# Adaptive Threshold System
# =============================================================================

@dataclass
class QualityThresholds:
    """Quality thresholds for alpha evaluation."""
    # Core metrics
    sharpe_min: float = 1.25
    sharpe_target: float = 1.58      # BRAIN pass threshold for delay-1
    fitness_min: float = 1.0
    turnover_max: float = 0.7
    
    # Advanced metrics
    os_sharpe_min: float = 0.5       # Out-of-sample minimum
    is_os_ratio_min: float = 0.4     # IS/OS Sharpe ratio minimum
    self_corr_max: float = 0.7       # Self-correlation maximum
    prod_corr_max: float = 0.65      # Production correlation maximum
    drawdown_max: float = 0.25       # Maximum drawdown
    
    # Optimization triggers
    optimize_sharpe_min: float = 0.3  # Minimum to consider optimization
    optimize_sharpe_max: float = 1.2  # Maximum (already good enough)
    
    # Region-specific adjustments
    region: str = "USA"
    adjustment_factor: float = 1.0
    
    def adjusted_sharpe_min(self) -> float:
        """Get adjusted Sharpe minimum based on region."""
        return self.sharpe_min * self.adjustment_factor
    
    def adjusted_sharpe_target(self) -> float:
        """Get adjusted Sharpe target based on region."""
        return self.sharpe_target * self.adjustment_factor
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "sharpe_min": self.sharpe_min,
            "sharpe_target": self.sharpe_target,
            "fitness_min": self.fitness_min,
            "turnover_max": self.turnover_max,
            "os_sharpe_min": self.os_sharpe_min,
            "is_os_ratio_min": self.is_os_ratio_min,
            "self_corr_max": self.self_corr_max,
            "prod_corr_max": self.prod_corr_max,
            "region": self.region,
            "adjustment_factor": self.adjustment_factor,
            "adjusted_sharpe_min": self.adjusted_sharpe_min(),
            "adjusted_sharpe_target": self.adjusted_sharpe_target(),
        }


# Region-specific threshold configurations
# Based on empirical observations and BRAIN platform requirements
REGION_THRESHOLDS = {
    "USA": QualityThresholds(
        sharpe_min=1.25,
        sharpe_target=1.58,
        fitness_min=1.0,
        turnover_max=0.7,
        os_sharpe_min=0.5,
        region="USA",
        adjustment_factor=1.0
    ),
    "EUR": QualityThresholds(
        sharpe_min=1.15,
        sharpe_target=1.50,
        fitness_min=0.9,
        turnover_max=0.65,
        os_sharpe_min=0.45,
        region="EUR",
        adjustment_factor=0.95
    ),
    "ASI": QualityThresholds(
        sharpe_min=1.0,
        sharpe_target=1.35,
        fitness_min=0.8,
        turnover_max=0.6,
        os_sharpe_min=0.4,
        region="ASI",
        adjustment_factor=0.85
    ),
    "KOR": QualityThresholds(
        sharpe_min=1.0,
        sharpe_target=1.35,
        fitness_min=0.8,
        turnover_max=0.6,
        os_sharpe_min=0.4,
        region="KOR",
        adjustment_factor=0.85
    ),
    "CHN": QualityThresholds(
        sharpe_min=1.1,
        sharpe_target=1.45,
        fitness_min=0.85,
        turnover_max=0.65,
        os_sharpe_min=0.45,
        region="CHN",
        adjustment_factor=0.9
    ),
    "IND": QualityThresholds(
        sharpe_min=0.9,
        sharpe_target=1.25,
        fitness_min=0.7,
        turnover_max=0.55,
        os_sharpe_min=0.35,
        region="IND",
        adjustment_factor=0.8
    ),
    "GLB": QualityThresholds(
        sharpe_min=1.0,
        sharpe_target=1.35,
        fitness_min=0.8,
        turnover_max=0.6,
        os_sharpe_min=0.4,
        region="GLB",
        adjustment_factor=0.85
    ),
}

# Dataset category adjustments (multiplied with region adjustment)
CATEGORY_ADJUSTMENTS = {
    "pv": 1.0,          # Standard for price-volume
    "analyst": 1.1,     # Analyst data often yields higher Sharpe
    "fundamental": 0.95, # Fundamental data slightly harder
    "news": 0.85,       # News/sentiment is noisier
    "other": 0.8,       # Alternative data more challenging
}


def get_thresholds(
    region: str = "USA",
    dataset_category: str = None,
    delay: int = 1
) -> QualityThresholds:
    """
    Get adaptive thresholds for a specific region and dataset category.
    
    Args:
        region: Market region (USA, EUR, ASI, KOR, CHN, IND, GLB)
        dataset_category: Optional dataset category for adjustment
        delay: Trading delay (0 or 1)
    
    Returns:
        QualityThresholds with appropriate adjustments
    """
    # Get base thresholds for region
    base = REGION_THRESHOLDS.get(region.upper(), REGION_THRESHOLDS["USA"])
    
    # Create copy with adjustments
    thresholds = QualityThresholds(
        sharpe_min=base.sharpe_min,
        sharpe_target=base.sharpe_target,
        fitness_min=base.fitness_min,
        turnover_max=base.turnover_max,
        os_sharpe_min=base.os_sharpe_min,
        is_os_ratio_min=base.is_os_ratio_min,
        self_corr_max=base.self_corr_max,
        prod_corr_max=base.prod_corr_max,
        optimize_sharpe_min=base.optimize_sharpe_min,
        optimize_sharpe_max=base.optimize_sharpe_max,
        region=region.upper(),
        adjustment_factor=base.adjustment_factor
    )
    
    # Apply category adjustment
    if dataset_category:
        cat_adj = CATEGORY_ADJUSTMENTS.get(
            dataset_category.lower(), 
            CATEGORY_ADJUSTMENTS["other"]
        )
        thresholds.adjustment_factor *= cat_adj
        
        # Recalculate targets
        thresholds.sharpe_min *= cat_adj
        thresholds.sharpe_target *= cat_adj
        thresholds.fitness_min *= cat_adj
    
    # Adjust for delay-0 (higher standards)
    if delay == 0:
        thresholds.sharpe_min *= 1.15
        thresholds.sharpe_target *= 1.15
        thresholds.fitness_min *= 1.1
        thresholds.turnover_max *= 0.9  # Lower turnover allowed
    
    logger.debug(
        f"[Thresholds] region={region} category={dataset_category} delay={delay} "
        f"-> sharpe_min={thresholds.sharpe_min:.2f} sharpe_target={thresholds.sharpe_target:.2f}"
    )
    
    return thresholds


@dataclass
class AlphaEvaluation:
    """Comprehensive alpha evaluation result."""
    # Pass/Fail
    passed: bool = False
    quality_status: str = "PENDING"  # PASS, REJECT, OPTIMIZE
    
    # Scores
    composite_score: float = 0.0
    sharpe_score: float = 0.0
    fitness_score: float = 0.0
    turnover_score: float = 0.0
    robustness_score: float = 0.0
    
    # Raw metrics
    is_sharpe: float = 0.0
    os_sharpe: float = 0.0
    fitness: float = 0.0
    turnover: float = 0.0
    drawdown: float = 0.0
    
    # Thresholds used
    thresholds: QualityThresholds = field(default_factory=QualityThresholds)
    
    # Failure reasons
    failed_tests: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "quality_status": self.quality_status,
            "composite_score": round(self.composite_score, 4),
            "sharpe_score": round(self.sharpe_score, 4),
            "fitness_score": round(self.fitness_score, 4),
            "turnover_score": round(self.turnover_score, 4),
            "robustness_score": round(self.robustness_score, 4),
            "is_sharpe": round(self.is_sharpe, 4),
            "os_sharpe": round(self.os_sharpe, 4),
            "fitness": round(self.fitness, 4),
            "turnover": round(self.turnover, 4),
            "drawdown": round(self.drawdown, 4),
            "failed_tests": self.failed_tests,
            "recommendations": self.recommendations,
        }


def evaluate_alpha_comprehensive(
    sim_result: Dict[str, Any],
    region: str = "USA",
    dataset_category: str = None,
    delay: int = 1,
    prod_corr: float = 0.0,
    self_corr: float = 0.0,
    use_brain_checks: bool = True  # 新增：是否使用 BRAIN 官方检查
) -> AlphaEvaluation:
    """
    Comprehensive alpha evaluation with adaptive thresholds.
    
    增强版：优先使用 BRAIN 平台返回的官方检查结果 (`checks` 数组)，
    这比硬编码阈值更准确，因为平台的阈值可能因 region/delay 而异。
    
    Args:
        sim_result: Simulation result from BRAIN API
        region: Market region
        dataset_category: Dataset category for threshold adjustment
        delay: Trading delay
        prod_corr: Production correlation
        self_corr: Self-correlation
        use_brain_checks: 是否优先使用 BRAIN 平台的官方检查结果
    
    Returns:
        AlphaEvaluation with detailed assessment
    """
    # Get adaptive thresholds (作为 fallback)
    thresholds = get_thresholds(region, dataset_category, delay)
    
    # 尝试从 BRAIN 检查结果中获取官方阈值
    if use_brain_checks:
        official_thresholds = get_official_thresholds_from_checks(sim_result)
        if official_thresholds:
            # 使用官方阈值覆盖
            if 'sharpe_min' in official_thresholds:
                thresholds.sharpe_target = official_thresholds['sharpe_min']
                thresholds.sharpe_min = official_thresholds['sharpe_min'] * 0.8
            if 'fitness_min' in official_thresholds:
                thresholds.fitness_min = official_thresholds['fitness_min']
            if 'turnover_max' in official_thresholds:
                thresholds.turnover_max = official_thresholds['turnover_max']
    
    # Extract metrics
    is_stats = _extract_is_stats(sim_result)
    os_stats = _extract_os_stats(sim_result)
    
    is_sharpe = _safe_float(is_stats.get('sharpe', is_stats.get('Sharpe', 0)))
    os_sharpe = _safe_float(os_stats.get('sharpe', os_stats.get('Sharpe', 0)))
    fitness = _safe_float(is_stats.get('fitness', is_stats.get('Fitness', 0)))
    turnover = _safe_float(is_stats.get('turnover', is_stats.get('Turnover', 0)))
    drawdown = _safe_float(is_stats.get('drawdown', is_stats.get('Drawdown', 0)))
    
    # 新增：提取额外的有用指标
    margin = _safe_float(is_stats.get('margin', 0))
    long_count = is_stats.get('longCount', 0)
    short_count = is_stats.get('shortCount', 0)
    
    # Create evaluation
    eval_result = AlphaEvaluation(
        is_sharpe=is_sharpe,
        os_sharpe=os_sharpe,
        fitness=fitness,
        turnover=turnover,
        drawdown=drawdown,
        thresholds=thresholds
    )
    
    # Score calculations (normalized 0-1)
    eval_result.sharpe_score = min(1.0, is_sharpe / thresholds.sharpe_target) if is_sharpe > 0 else 0
    eval_result.fitness_score = min(1.0, fitness / thresholds.fitness_min) if fitness > 0 else 0
    eval_result.turnover_score = max(0, 1.0 - turnover / thresholds.turnover_max) if turnover < thresholds.turnover_max else 0
    
    # Robustness: IS/OS consistency
    if is_sharpe > 0 and os_sharpe > 0:
        eval_result.robustness_score = min(1.0, (os_sharpe / is_sharpe) / thresholds.is_os_ratio_min)
    elif os_sharpe > 0:
        eval_result.robustness_score = 0.5  # OOS positive but IS not
    else:
        eval_result.robustness_score = 0.0
    
    # Composite score
    eval_result.composite_score = (
        0.40 * eval_result.sharpe_score +
        0.25 * eval_result.fitness_score +
        0.15 * eval_result.turnover_score +
        0.20 * eval_result.robustness_score
    )
    
    # =========================================================================
    # 使用 BRAIN 官方检查结果（如果可用）
    # =========================================================================
    failed_tests = []
    recommendations = []
    
    if use_brain_checks:
        brain_eval = evaluate_with_brain_checks(sim_result)
        
        # 直接使用平台的检查结果
        if brain_eval['check_details']:
            for check in brain_eval['check_details']:
                check_name = check.get('name', '')
                check_result = check.get('result', 'PENDING')
                check_limit = check.get('limit')
                check_value = check.get('value')
                
                if check_result == 'FAIL':
                    # 构建详细的失败信息
                    if check_limit is not None and check_value is not None:
                        failed_tests.append(f"{check_name} (value={check_value:.2f}, limit={check_limit:.2f})")
                    else:
                        failed_tests.append(check_name)
                    
                    # 根据检查类型给出建议
                    recommendations.extend(_get_recommendations_for_check(check_name))
            
            # 如果平台已经告诉我们可以提交，直接使用
            if brain_eval['can_submit'] and not failed_tests:
                eval_result.passed = True
                eval_result.quality_status = "PASS"
                eval_result.failed_tests = []
                eval_result.recommendations = []
                return eval_result
    
    # =========================================================================
    # Fallback: 使用本地阈值判断（当 BRAIN checks 不可用时）
    # =========================================================================
    if not failed_tests:  # 只有在没有从 checks 获取到失败信息时才使用本地判断
        # Test 1: Sharpe
        if is_sharpe < thresholds.adjusted_sharpe_min():
            failed_tests.append(f"LOW_SHARPE (is={is_sharpe:.2f} < {thresholds.adjusted_sharpe_min():.2f})")
            if is_sharpe > 0:
                recommendations.append("Consider adding decay or smoothing operators")
        
        # Test 2: Fitness
        if fitness < thresholds.fitness_min:
            failed_tests.append(f"LOW_FITNESS (fit={fitness:.2f} < {thresholds.fitness_min:.2f})")
            recommendations.append("Try different neutralization or add risk controls")
        
        # Test 3: Turnover
        if turnover > thresholds.turnover_max:
            failed_tests.append(f"HIGH_TURNOVER (to={turnover:.2f} > {thresholds.turnover_max:.2f})")
            recommendations.append("Add ts_decay_linear or increase lookback windows")
        
        # Test 4: OS performance
        if is_sharpe > 0.5 and os_sharpe < thresholds.os_sharpe_min:
            failed_tests.append(f"POOR_OOS (os={os_sharpe:.2f} < {thresholds.os_sharpe_min:.2f})")
            recommendations.append("Possible overfitting - simplify expression or add smoothing")
        
        # Test 5: IS/OS ratio
        if is_sharpe > 0 and (os_sharpe / (is_sharpe + 1e-9)) < thresholds.is_os_ratio_min:
            failed_tests.append(f"OVERFITTING (is/os ratio too low)")
            recommendations.append("Reduce complexity or add more regularization")
    
    # 相关性检查（总是使用传入的值）
    if self_corr > thresholds.self_corr_max:
        failed_tests.append(f"HIGH_SELF_CORR (sc={self_corr:.2f} > {thresholds.self_corr_max:.2f})")
        recommendations.append("Expression too similar to existing alphas")
    
    if prod_corr > thresholds.prod_corr_max:
        failed_tests.append(f"HIGH_PROD_CORR (pc={prod_corr:.2f} > {thresholds.prod_corr_max:.2f})")
        recommendations.append("Try different fields or operators for more novelty")
    
    # 新增：权重集中度检查
    if long_count and short_count:
        total_positions = long_count + short_count
        if total_positions < 10:  # 持仓过于集中
            failed_tests.append(f"CONCENTRATED_WEIGHT (positions={total_positions})")
            recommendations.append("Consider using rank() or grouping operators for better diversification")
    
    eval_result.failed_tests = failed_tests
    eval_result.recommendations = list(set(recommendations))  # 去重
    
    # Determine quality status
    if not failed_tests:
        eval_result.passed = True
        eval_result.quality_status = "PASS"
    elif is_sharpe >= thresholds.optimize_sharpe_min and is_sharpe < thresholds.optimize_sharpe_max:
        # Potentially optimizable
        eval_result.quality_status = "OPTIMIZE"
    else:
        eval_result.quality_status = "REJECT"
    
    logger.debug(
        f"[AlphaEval] sharpe={is_sharpe:.2f}/{thresholds.adjusted_sharpe_min():.2f} "
        f"fitness={fitness:.2f} turnover={turnover:.2f} -> {eval_result.quality_status}"
    )
    
    return eval_result


def _get_recommendations_for_check(check_name: str) -> List[str]:
    """根据检查类型返回优化建议。"""
    recommendations_map = {
        'LOW_SHARPE': ["Consider adding decay or smoothing operators", "Try different neutralization settings"],
        'LOW_FITNESS': ["Adjust neutralization method", "Add risk control operators"],
        'HIGH_TURNOVER': ["Add ts_decay_linear or increase lookback windows", "Use slower-changing signals"],
        'LOW_TURNOVER': ["Reduce smoothing/decay", "Use more responsive signals"],
        'CONCENTRATED_WEIGHT': ["Use rank() for better weight distribution", "Apply truncation or winsorize"],
        'LOW_SUB_UNIVERSE_SHARPE': ["Test on different sub-universes", "Check for sector/size bias"],
        'LOW_2Y_SHARPE': ["Check for regime dependency", "Test robustness across time periods"],
        'LOW_INVESTABILITY_CONSTRAINED_SHARPE': ["Reduce position concentration", "Add liquidity filters"],
        'SELF_CORRELATION': ["Modify expression structure", "Use different operators or fields"],
        'PROD_CORRELATION': ["Try different data fields", "Use novel operator combinations"],
    }
    return recommendations_map.get(check_name, [])


def calculate_alpha_score(
    sim_result: Dict[str, Any],
    prod_corr: float = 0.0,
    self_corr: float = 0.0,
    weights: Optional[Dict[str, float]] = None
) -> float:
    """
    Calculate composite alpha score from simulation results.
    
    Score = w_test * S_test + w_train * S_train + w_fitness * Fitness
            - w_corr * max(0, prod_corr - 0.7)
            - w_turnover * turnover_penalty
            - w_invest * investability_penalty
    
    Args:
        sim_result: Simulation result dictionary from Brain API
        prod_corr: Maximum correlation with production alphas (0-1)
        self_corr: Self-correlation value (0-1)
        weights: Optional custom weights, defaults to optimized values
    
    Returns:
        Composite score (higher is better)
    """
    # Default weights based on 优化.md recommendations
    default_weights = {
        'test_sharpe': 0.55,
        'train_sharpe': 0.25,
        'fitness': 0.20,
        'prod_corr_penalty': 0.30,
        'turnover_penalty': 0.15,
        'investability_penalty': 0.20,
    }
    w = weights or default_weights
    
    # Extract metrics with safe defaults
    is_stats = _extract_is_stats(sim_result)
    os_stats = _extract_os_stats(sim_result)
    
    # Core performance metrics (note: lowercase keys in actual data)
    test_sharpe = _safe_float(os_stats.get('sharpe', os_stats.get('Sharpe', 0.0)))
    train_sharpe = _safe_float(is_stats.get('sharpe', is_stats.get('Sharpe', 0.0)))
    fitness = _safe_float(is_stats.get('fitness', is_stats.get('Fitness', 0.0)))
    
    # Risk/constraint metrics
    turnover = _safe_float(is_stats.get('turnover', is_stats.get('Turnover', 0.0)))
    
    # Investability-constrained metrics
    invest_constrained = _extract_investability_stats(sim_result)
    invest_sharpe = _safe_float(invest_constrained.get('sharpe', invest_constrained.get('Sharpe', train_sharpe)))
    
    # Calculate penalties
    corr_penalty = max(0, prod_corr - 0.7)
    
    # Turnover penalty: penalize high turnover (> 50%)
    turnover_penalty = max(0, turnover - 0.5) if turnover else 0.0
    
    # Investability penalty: difference between raw and constrained Sharpe
    investability_penalty = max(0, train_sharpe - invest_sharpe) if invest_sharpe else 0.0
    
    # Calculate composite score
    score = (
        w['test_sharpe'] * test_sharpe +
        w['train_sharpe'] * train_sharpe +
        w['fitness'] * fitness -
        w['prod_corr_penalty'] * corr_penalty -
        w['turnover_penalty'] * turnover_penalty -
        w['investability_penalty'] * investability_penalty
    )
    
    logger.debug(
        f"得分明细: 测试集={test_sharpe:.3f}, 训练集={train_sharpe:.3f}, "
        f"Fitness={fitness:.3f}, 相关性惩罚={corr_penalty:.3f}, "
        f"换手惩罚={turnover_penalty:.3f}, 可投资性惩罚={investability_penalty:.3f} -> 总分 {score:.3f}"
    )
    
    return score


def _extract_is_stats(sim_result: Dict) -> Dict:
    """从模拟结果中提取训练集统计信息。"""
    # Try multiple possible locations based on actual ace_lib output
    # Priority: train -> is_stats[0] -> is
    if 'train' in sim_result and sim_result['train']:
        return sim_result['train']
    if 'is_stats' in sim_result:
        is_stats = sim_result['is_stats']
        if isinstance(is_stats, list) and len(is_stats) > 0:
            return is_stats[0]
    if 'is' in sim_result:
        return sim_result['is'] or {}
    if 'pnl' in sim_result and isinstance(sim_result['pnl'], dict):
        return sim_result['pnl'].get('is', {}) or {}
    return {}


def _extract_os_stats(sim_result: Dict) -> Dict:
    """从模拟结果中提取测试集统计信息。"""
    # Priority: test -> os
    if 'test' in sim_result and sim_result['test']:
        return sim_result['test']
    if 'os' in sim_result:
        return sim_result['os'] or {}
    if 'pnl' in sim_result and isinstance(sim_result['pnl'], dict):
        return sim_result['pnl'].get('os', {}) or {}
    return {}


def _extract_investability_stats(sim_result: Dict) -> Dict:
    """提取可投资性受限的统计信息。"""
    # Check in train stats first
    train_stats = _extract_is_stats(sim_result)
    if 'investabilityConstrained' in train_stats:
        return train_stats['investabilityConstrained'] or {}
    # Then check top-level
    if 'investabilityConstrained' in sim_result:
        return sim_result['investabilityConstrained'] or {}
    return {}


def _safe_float(value: Any) -> float:
    """安全地转换为浮点数，失败则返回 0.0。"""
    if value is None:
        return 0.0
    try:
        return float(value)
    except (ValueError, TypeError):
        return 0.0


def evaluate_alpha_tests(sim_result: Dict) -> Dict[str, bool]:
    """
    评估 Alpha 通过了哪些测试。
    
    优先使用 BRAIN API 返回的 `checks` 数组（真实平台检查结果）。
    
    Returns:
        字典: 测试名称: 是否通过 (True/False)
    """
    # 优先从 is.checks 或 checks 获取真实的平台检查结果
    checks = _extract_brain_checks(sim_result)
    if checks:
        results = {}
        for check in checks:
            name = check.get('name', '')
            result = check.get('result', 'PENDING')
            # PASS = 通过, FAIL/WARNING/PENDING = 不通过
            results[name] = (result == 'PASS')
        return results
    
    # Fallback: 旧格式兼容
    tests = sim_result.get('tests', {})
    if not tests:
        return {}
    
    results = {}
    for test_name, test_result in tests.items():
        if isinstance(test_result, dict):
            results[test_name] = test_result.get('result') == 'PASS'
        elif isinstance(test_result, str):
            results[test_name] = test_result == 'PASS'
        else:
            results[test_name] = bool(test_result)
    
    return results


def get_failed_tests(sim_result: Dict) -> list:
    """获取 Alpha 未通过的测试列表。"""
    test_results = evaluate_alpha_tests(sim_result)
    return [name for name, passed in test_results.items() if not passed]


def _extract_brain_checks(sim_result: Dict) -> List[Dict]:
    """
    从模拟结果中提取 BRAIN 平台的官方检查结果。
    
    真实 API 返回的 checks 结构:
    [
        {"name": "LOW_SHARPE", "result": "FAIL", "limit": 1.58, "value": -0.26},
        {"name": "HIGH_TURNOVER", "result": "PASS", "limit": 0.7, "value": 0.5},
        {"name": "SELF_CORRELATION", "result": "PENDING"},
        ...
    ]
    """
    # 优先从 is.checks 获取
    is_stats = sim_result.get('is', {}) or {}
    if 'checks' in is_stats:
        return is_stats.get('checks', [])
    
    # 尝试从顶层获取
    if 'checks' in sim_result:
        return sim_result.get('checks', [])
    
    return []


def evaluate_with_brain_checks(sim_result: Dict) -> Dict[str, Any]:
    """
    使用 BRAIN 平台返回的官方检查结果进行评估。
    
    这比使用硬编码阈值更准确，因为直接使用平台的真实阈值和判断。
    
    Returns:
        {
            'can_submit': bool,  # 是否可以提交
            'passed_checks': List[str],
            'failed_checks': List[str],
            'pending_checks': List[str],
            'warning_checks': List[str],
            'check_details': List[Dict],  # 完整的检查详情
            'pyramid_info': Dict,  # 金字塔乘数信息
            'competition_info': Dict,  # 比赛匹配信息
        }
    """
    checks = _extract_brain_checks(sim_result)
    
    passed = []
    failed = []
    pending = []
    warnings = []
    
    pyramid_info = {}
    competition_info = {}
    
    for check in checks:
        name = check.get('name', '')
        result = check.get('result', 'PENDING')
        
        if result == 'PASS':
            passed.append(name)
        elif result == 'FAIL':
            failed.append(name)
        elif result == 'WARNING':
            warnings.append(name)
        else:  # PENDING or unknown
            pending.append(name)
        
        # 提取金字塔信息
        if name == 'MATCHES_PYRAMID':
            pyramid_info = {
                'multiplier': check.get('multiplier', 1.0),
                'effective': check.get('effective', 1),
                'pyramids': check.get('pyramids', []),
            }
        
        # 提取比赛信息
        if name == 'MATCHES_COMPETITION':
            competition_info = {
                'competitions': check.get('competitions', []),
            }
    
    # 直接使用 API 返回的 can_submit 或根据检查结果推断
    can_submit = sim_result.get('can_submit', False)
    if not can_submit and not failed and not pending:
        can_submit = True
    
    return {
        'can_submit': can_submit,
        'passed_checks': passed,
        'failed_checks': failed,
        'pending_checks': pending,
        'warning_checks': warnings,
        'check_details': checks,
        'pyramid_info': pyramid_info,
        'competition_info': competition_info,
    }


def get_official_thresholds_from_checks(sim_result: Dict) -> Dict[str, float]:
    """
    从 BRAIN 检查结果中提取官方阈值。
    
    这比硬编码更准确，因为不同 region/delay 的阈值可能不同。
    
    Returns:
        {
            'sharpe_min': 1.58,
            'fitness_min': 1.0,
            'turnover_max': 0.7,
            ...
        }
    """
    checks = _extract_brain_checks(sim_result)
    thresholds = {}
    
    threshold_mapping = {
        'LOW_SHARPE': 'sharpe_min',
        'LOW_FITNESS': 'fitness_min',
        'HIGH_TURNOVER': 'turnover_max',
        'LOW_TURNOVER': 'turnover_min',
        'CONCENTRATED_WEIGHT': 'weight_concentration_max',
        'LOW_SUB_UNIVERSE_SHARPE': 'sub_universe_sharpe_min',
        'LOW_2Y_SHARPE': 'sharpe_2y_min',
        'LOW_INVESTABILITY_CONSTRAINED_SHARPE': 'invest_sharpe_min',
    }
    
    for check in checks:
        name = check.get('name', '')
        limit = check.get('limit')
        
        if name in threshold_mapping and limit is not None:
            thresholds[threshold_mapping[name]] = limit
    
    return thresholds


def should_optimize(sim_result: Dict) -> Tuple[bool, str]:
    is_stats = _extract_is_stats(sim_result) or {}
    os_stats = _extract_os_stats(sim_result) or {}
    invest_stats = _extract_investability_stats(sim_result) or {}

    train_sharpe = _safe_float(is_stats.get('sharpe', is_stats.get('Sharpe', 0)))
    train_fitness = _safe_float(is_stats.get('fitness', is_stats.get('Fitness', 0)))
    train_turnover = _safe_float(is_stats.get('turnover', is_stats.get('Turnover', 0)))

    test_sharpe = _safe_float(os_stats.get('sharpe', os_stats.get('Sharpe', 0)))
    test_fitness = _safe_float(os_stats.get('fitness', os_stats.get('Fitness', 0)))

    invest_sharpe = _safe_float(invest_stats.get('sharpe', invest_stats.get('Sharpe', train_sharpe)))

    risk_neutral = sim_result.get('riskNeutralized', {}) or {}
    rn_sharpe = _safe_float(risk_neutral.get('sharpe', risk_neutral.get('Sharpe', train_sharpe)))

    # ---- 0) Fast reject: clearly bad / noisy ----
    # Negative in both IS and OOS: usually not worth 100-budget optimization
    if train_sharpe <= 0 and test_sharpe <= 0:
        return False, "IS/OOS均为负，淘汰"

    # Very weak + not rescued by RN: low ROI to optimize
    if train_sharpe < 0.15 and rn_sharpe < 0.4:
        return False, "信号过弱且风险中性化未救回，淘汰"

    # ---- 1) Already good (prefer tests-based if you have it) ----
    # If you can access pass/fail tests, check them here instead of hardcoding.
    if train_sharpe >= 1.58 and train_fitness >= 1.0:
        # still sanity-check OOS
        if test_sharpe >= 0.8:
            return False, "已接近/达到门槛且OOS不差，跳过优化"
        # else: good IS but weak OOS -> optimize for robustness
        return True, "IS达标但OOS偏弱，做稳健性优化"

    # ---- 2) High-value optimization triggers (fixable failure modes) ----
    # A) Risk exposure issue: RN improves a lot
    if (rn_sharpe - train_sharpe) >= 0.25 and rn_sharpe >= 0.6:
        return True, "风险中性化显著改善：优先调neutralization/结构去风险"

    # B) Investability drops a lot
    if (train_sharpe - invest_sharpe) >= 0.25 and train_sharpe >= 0.3:
        return True, "可投资性约束下掉得多：优先降集中/做更强归一化/更平滑"

    # C) Overfitting: big IS→OOS gap
    if train_sharpe >= 0.4:
        ratio = test_sharpe / (train_sharpe + 1e-9)
        gap = train_sharpe - test_sharpe
        if ratio < 0.5 and gap >= 0.3:
            return True, "IS→OOS衰减明显：优先加平滑/增大窗口/提高decay"

    # D) Turnover too extreme (if you have a target band)
    # Here we only trigger if it's extremely high/low; avoid over-filtering.
    if train_turnover > 0.6:
        return True, "换手过高：优先增大窗口/加decay/改更平滑结构"

    # ---- 3) The sweet spot: positive but not yet passing ----
    # This is where optimization pays off the most.
    if 0.15 <= train_sharpe < 1.58:
        # If OOS isn't catastrophic, worth optimizing
        if test_sharpe > -0.2 and test_fitness > -0.2:
            return True, "正信号但未达标：做窗口/标准化/settings小扫"
        # If OOS very bad, only optimize if RN rescues (already handled above)
        return False, "OOS过差且无救回迹象，淘汰"

    # ---- 4) Default ----
    return True, "默认：可尝试低成本优化"
