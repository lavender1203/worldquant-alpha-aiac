"""
Brain Protocol - Abstract interface for WorldQuant BRAIN platform integration

This protocol defines the contract for interacting with the BRAIN API,
allowing for easy testing with mock implementations.
"""

from typing import Protocol, List, Dict, Any, Optional, runtime_checkable
from dataclasses import dataclass, field


@dataclass
class SimulationSettings:
    """Settings for alpha simulation."""
    region: str = "USA"
    universe: str = "TOP3000"
    delay: int = 1
    decay: int = 4
    neutralization: str = "SUBINDUSTRY"
    truncation: float = 0.08
    test_period: str = "P2Y0M"
    nan_handling: str = "OFF"
    unit_handling: str = "VERIFY"
    pasteurization: str = "ON"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to API payload format."""
        return {
            "instrumentType": "EQUITY",
            "region": self.region,
            "universe": self.universe,
            "delay": self.delay,
            "decay": self.decay,
            "neutralization": self.neutralization,
            "truncation": self.truncation,
            "testPeriod": self.test_period,
            "nanHandling": self.nan_handling,
            "unitHandling": self.unit_handling,
            "pasteurization": self.pasteurization,
            "language": "FASTEXPR",
            "visualization": False,
        }


@dataclass
class SimulationMetrics:
    """
    Metrics returned from a simulation.
    
    Real API structure from BRAIN platform:
    - is/train/test/os each contain: pnl, bookSize, longCount, shortCount,
      turnover, returns, drawdown, margin, sharpe, fitness, startDate,
      investabilityConstrained{}, riskNeutralized{}
    """
    # Primary IS metrics
    sharpe: Optional[float] = None
    fitness: Optional[float] = None
    turnover: Optional[float] = None
    returns: Optional[float] = None
    drawdown: Optional[float] = None  # API uses 'drawdown', not 'max_dd'
    pnl: Optional[float] = None
    margin: Optional[float] = None
    bookSize: Optional[float] = None
    longCount: Optional[int] = None
    shortCount: Optional[int] = None
    
    # Train period metrics
    train_sharpe: Optional[float] = None
    train_fitness: Optional[float] = None
    train_turnover: Optional[float] = None
    train_returns: Optional[float] = None
    train_drawdown: Optional[float] = None
    
    # Test period metrics
    test_sharpe: Optional[float] = None
    test_fitness: Optional[float] = None
    test_turnover: Optional[float] = None
    test_returns: Optional[float] = None
    test_drawdown: Optional[float] = None
    
    # Out-of-sample metrics
    os_sharpe: Optional[float] = None
    os_fitness: Optional[float] = None
    
    # Nested constraint metrics (API returns these as nested objects)
    investability_constrained: Dict[str, Any] = field(default_factory=dict)
    risk_neutralized: Dict[str, Any] = field(default_factory=dict)
    train_investability_constrained: Dict[str, Any] = field(default_factory=dict)
    train_risk_neutralized: Dict[str, Any] = field(default_factory=dict)
    test_investability_constrained: Dict[str, Any] = field(default_factory=dict)
    test_risk_neutralized: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SubmissionCheck:
    """
    A submission check result from BRAIN platform.
    
    Real API structure:
    - name: Check name (e.g., LOW_SHARPE, HIGH_TURNOVER, SELF_CORRELATION)
    - result: PASS, FAIL, PENDING, WARNING
    - limit: Threshold value (optional)
    - value: Actual value (optional)
    """
    name: str
    result: str  # PASS, FAIL, PENDING, WARNING
    limit: Optional[float] = None
    value: Optional[float] = None
    date: Optional[str] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SubmissionCheck":
        return cls(
            name=data.get("name", ""),
            result=data.get("result", "PENDING"),
            limit=data.get("limit"),
            value=data.get("value"),
            date=data.get("date"),
        )


@dataclass
class SimulationResult:
    """
    Result of an alpha simulation.
    
    Real API structure includes:
    - id, type, author, settings, regular{code, description, operatorCount}
    - stage (IS/OS), status (UNSUBMITTED/SUBMITTED)
    - is, train, test, os - each with full metrics
    - checks - list of submission validation checks
    """
    success: bool
    alpha_id: Optional[str] = None
    expression: Optional[str] = None
    settings: Optional[Dict[str, Any]] = None
    metrics: Optional[SimulationMetrics] = None
    error_message: Optional[str] = None
    
    # Alpha metadata
    stage: Optional[str] = None  # IS or OS
    status: Optional[str] = None  # UNSUBMITTED, SUBMITTED, etc.
    alpha_type: Optional[str] = None  # REGULAR or SUPER
    date_created: Optional[str] = None
    date_submitted: Optional[str] = None
    
    # Submission checks (critical for knowing if alpha can be submitted)
    checks: List[SubmissionCheck] = field(default_factory=list)
    failed_checks: List[str] = field(default_factory=list)
    pending_checks: List[str] = field(default_factory=list)
    passed_checks: List[str] = field(default_factory=list)
    can_submit: bool = False
    
    # Raw data for debugging
    is_data: Optional[Dict[str, Any]] = None  # In-sample data
    os_data: Optional[Dict[str, Any]] = None  # Out-of-sample data
    train_data: Optional[Dict[str, Any]] = None
    test_data: Optional[Dict[str, Any]] = None
    raw: Optional[Dict[str, Any]] = None
    
    @classmethod
    def from_error(cls, error: str) -> "SimulationResult":
        """Create a failed result from an error message."""
        return cls(success=False, error_message=error)
    
    @classmethod
    def from_api_response(cls, data: Dict[str, Any]) -> "SimulationResult":
        """Create a result from the raw API response."""
        if not data.get("success", False):
            return cls(success=False, error_message=data.get("error", "Unknown error"))
        
        # Parse metrics from API response
        is_data = data.get("is", {})
        os_data = data.get("os", {})
        train_data = data.get("train", {})
        test_data = data.get("test", {})
        metrics_raw = data.get("metrics", {})
        
        metrics = SimulationMetrics(
            sharpe=metrics_raw.get("sharpe") or is_data.get("sharpe"),
            fitness=metrics_raw.get("fitness") or is_data.get("fitness"),
            turnover=metrics_raw.get("turnover") or is_data.get("turnover"),
            returns=metrics_raw.get("returns") or is_data.get("returns"),
            drawdown=metrics_raw.get("drawdown") or is_data.get("drawdown"),
            pnl=metrics_raw.get("pnl") or is_data.get("pnl"),
            margin=metrics_raw.get("margin") or is_data.get("margin"),
            bookSize=metrics_raw.get("bookSize") or is_data.get("bookSize"),
            longCount=metrics_raw.get("longCount") or is_data.get("longCount"),
            shortCount=metrics_raw.get("shortCount") or is_data.get("shortCount"),
            train_sharpe=metrics_raw.get("train_sharpe") or (train_data.get("sharpe") if train_data else None),
            train_fitness=metrics_raw.get("train_fitness") or (train_data.get("fitness") if train_data else None),
            train_turnover=metrics_raw.get("train_turnover") or (train_data.get("turnover") if train_data else None),
            train_returns=metrics_raw.get("train_returns") or (train_data.get("returns") if train_data else None),
            train_drawdown=metrics_raw.get("train_drawdown") or (train_data.get("drawdown") if train_data else None),
            test_sharpe=metrics_raw.get("test_sharpe") or (test_data.get("sharpe") if test_data else None),
            test_fitness=metrics_raw.get("test_fitness") or (test_data.get("fitness") if test_data else None),
            test_turnover=metrics_raw.get("test_turnover") or (test_data.get("turnover") if test_data else None),
            test_returns=metrics_raw.get("test_returns") or (test_data.get("returns") if test_data else None),
            test_drawdown=metrics_raw.get("test_drawdown") or (test_data.get("drawdown") if test_data else None),
            os_sharpe=metrics_raw.get("os_sharpe") or (os_data.get("sharpe") if os_data else None),
            os_fitness=metrics_raw.get("os_fitness") or (os_data.get("fitness") if os_data else None),
            investability_constrained=metrics_raw.get("investabilityConstrained", {}),
            risk_neutralized=metrics_raw.get("riskNeutralized", {}),
            train_investability_constrained=metrics_raw.get("train_investabilityConstrained", {}),
            train_risk_neutralized=metrics_raw.get("train_riskNeutralized", {}),
            test_investability_constrained=metrics_raw.get("test_investabilityConstrained", {}),
            test_risk_neutralized=metrics_raw.get("test_riskNeutralized", {}),
        )
        
        # Parse checks
        raw_checks = data.get("checks", [])
        checks = [SubmissionCheck.from_dict(c) for c in raw_checks]
        failed_checks = data.get("failed_checks", [])
        pending_checks = data.get("pending_checks", [])
        passed_checks = data.get("passed_checks", [])
        can_submit = data.get("can_submit", False)
        
        return cls(
            success=True,
            alpha_id=data.get("alpha_id"),
            expression=data.get("expression"),
            settings=data.get("settings"),
            metrics=metrics,
            stage=data.get("stage"),
            status=data.get("status"),
            alpha_type=data.get("type"),
            date_created=data.get("dateCreated"),
            date_submitted=data.get("dateSubmitted"),
            checks=checks,
            failed_checks=failed_checks,
            pending_checks=pending_checks,
            passed_checks=passed_checks,
            can_submit=can_submit,
            is_data=is_data,
            os_data=os_data,
            train_data=train_data,
            test_data=test_data,
            raw=data.get("raw"),
        )


@dataclass
class DatasetInfo:
    """
    Information about a BRAIN dataset.
    
    Real API structure from get_datasets:
    - id, name (nested in dataset object for fields)
    - category, subcategory, description
    - coverage, dateCoverage
    - themes, resources, pyramidMultiplier
    """
    id: str
    name: str
    description: Optional[str] = None
    category: Optional[str] = None
    subcategory: Optional[str] = None
    coverage: Optional[float] = None
    date_coverage: Optional[float] = None
    field_count: Optional[int] = None
    pyramid_multiplier: Optional[float] = None
    themes: List[Dict[str, Any]] = field(default_factory=list)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetInfo":
        return cls(
            id=data.get("id", ""),
            name=data.get("name", ""),
            description=data.get("description"),
            category=data.get("category"),
            subcategory=data.get("subcategory"),
            coverage=data.get("coverage"),
            date_coverage=data.get("dateCoverage"),
            pyramid_multiplier=data.get("pyramidMultiplier"),
            themes=data.get("themes", []),
        )


@dataclass
class DataFieldInfo:
    """
    Information about a BRAIN data field.
    
    Real API structure from get_datafields:
    - id: field ID (e.g., "assets", "close")
    - description: field description
    - dataset: nested object {id, name}
    - category: nested object {id, name}
    - subcategory: nested object {id, name}
    - region, delay, universe
    - type: MATRIX, VECTOR, GROUP
    - dateCoverage, coverage
    - userCount, alphaCount
    - pyramidMultiplier
    - themes: list of theme objects
    """
    id: str
    description: Optional[str] = None
    dataset_id: Optional[str] = None
    dataset_name: Optional[str] = None
    category_id: Optional[str] = None
    category_name: Optional[str] = None
    subcategory_id: Optional[str] = None
    subcategory_name: Optional[str] = None
    region: Optional[str] = None
    delay: Optional[int] = None
    universe: Optional[str] = None
    type: Optional[str] = None  # MATRIX, VECTOR, GROUP
    date_coverage: Optional[float] = None
    coverage: Optional[float] = None
    user_count: Optional[int] = None
    alpha_count: Optional[int] = None
    pyramid_multiplier: Optional[float] = None
    themes: List[Dict[str, Any]] = field(default_factory=list)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DataFieldInfo":
        """Create from API response dict."""
        dataset = data.get("dataset", {}) or {}
        category = data.get("category", {}) or {}
        subcategory = data.get("subcategory", {}) or {}
        
        return cls(
            id=data.get("id", ""),
            description=data.get("description"),
            dataset_id=dataset.get("id"),
            dataset_name=dataset.get("name"),
            category_id=category.get("id"),
            category_name=category.get("name"),
            subcategory_id=subcategory.get("id"),
            subcategory_name=subcategory.get("name"),
            region=data.get("region"),
            delay=data.get("delay"),
            universe=data.get("universe"),
            type=data.get("type"),
            date_coverage=data.get("dateCoverage"),
            coverage=data.get("coverage"),
            user_count=data.get("userCount"),
            alpha_count=data.get("alphaCount"),
            pyramid_multiplier=data.get("pyramidMultiplier"),
            themes=data.get("themes", []),
        )


@dataclass
class OperatorInfo:
    """
    Information about a BRAIN operator.
    
    Real API structure from get_operators:
    - name: operator name (e.g., "ts_rank", "add")
    - category: operator category (e.g., "Arithmetic", "Time Series")
    - scope: list of scopes ["COMBO", "REGULAR", "SELECTION"]
    - definition: usage definition (e.g., "ts_rank(x, d)")
    - description: detailed description
    - documentation: documentation URL path
    - level: operator level
    """
    name: str
    category: Optional[str] = None
    scope: List[str] = field(default_factory=list)
    definition: Optional[str] = None
    description: Optional[str] = None
    documentation: Optional[str] = None
    level: Optional[str] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OperatorInfo":
        return cls(
            name=data.get("name", ""),
            category=data.get("category"),
            scope=data.get("scope", []),
            definition=data.get("definition"),
            description=data.get("description"),
            documentation=data.get("documentation"),
            level=data.get("level"),
        )


@runtime_checkable
class BrainProtocol(Protocol):
    """
    Protocol for BRAIN platform API interactions.
    
    Implementations must provide methods for:
    - Authentication and session management
    - Alpha simulation (single and batch)
    - Dataset and datafield queries
    - Operator information
    """
    
    async def ensure_session(self) -> None:
        """Ensure a valid authenticated session exists."""
        ...
    
    async def authenticate(self) -> bool:
        """
        Authenticate with the BRAIN platform.
        
        Returns:
            True if authentication successful, False otherwise
        """
        ...
    
    async def simulate_alpha(
        self,
        expression: str,
        region: str = "USA",
        universe: str = "TOP3000",
        delay: int = 1,
        decay: int = 4,
        neutralization: str = "SUBINDUSTRY",
        truncation: float = 0.08,
        test_period: str = "P2Y0M",
    ) -> Dict[str, Any]:
        """
        Simulate a single alpha expression.
        
        Args:
            expression: The alpha expression to simulate
            region: Market region (USA, EUR, ASI, etc.)
            universe: Stock universe (TOP3000, TOP1000, etc.)
            delay: Signal delay in days
            decay: Decay factor
            neutralization: Neutralization method
            truncation: Truncation factor
            test_period: Test period in ISO 8601 duration format
            
        Returns:
            Dict containing simulation results with success status and metrics
        """
        ...
    
    async def simulate_batch(
        self,
        expressions: List[str],
        region: str = "USA",
        universe: str = "TOP3000",
        delay: int = 1,
        decay: int = 4,
        neutralization: str = "SUBINDUSTRY",
        truncation: float = 0.08,
        test_period: str = "P2Y0M",
        max_wait: int = 1200,
        timeout_grace_seconds: int = 180,
        no_child_timeout_seconds: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Simulate multiple alpha expressions in batch.
        
        Args:
            expressions: List of alpha expressions to simulate
            region: Market region
            universe: Stock universe
            delay: Signal delay in days
            decay: Decay factor
            neutralization: Neutralization method
            truncation: Truncation factor
            test_period: Test period
            
        Returns:
            List of simulation results in the same order as input expressions
        """
        ...
    
    async def get_datasets(
        self,
        region: str = "USA",
        delay: int = 1,
        universe: str = "TOP3000",
    ) -> List[Dict[str, Any]]:
        """
        Get available datasets for the specified region and settings.
        
        Args:
            region: Market region
            delay: Signal delay
            universe: Stock universe
            
        Returns:
            List of dataset information dictionaries
        """
        ...
    
    async def get_datafields(
        self,
        dataset_id: str,
        region: str = "USA",
        delay: int = 1,
        universe: str = "TOP3000",
    ) -> List[Dict[str, Any]]:
        """
        Get data fields for a specific dataset.
        
        Args:
            dataset_id: The dataset ID to query
            region: Market region
            delay: Signal delay
            universe: Stock universe
            
        Returns:
            List of data field information dictionaries
        """
        ...
    
    async def get_operators(self, detailed: bool = False) -> List[Any]:
        """
        Get available operators.
        
        Args:
            detailed: If True, return full operator info; if False, return names only
            
        Returns:
            List of operator names or operator info dictionaries
        """
        ...
    
    async def get_alpha_pnl(self, alpha_id: str) -> Dict[str, Any]:
        """
        Get PnL data for an alpha.
        
        Args:
            alpha_id: The alpha ID
            
        Returns:
            PnL data dictionary
        """
        ...
    
    async def check_correlation(
        self,
        alpha_id: str,
        check_type: str = "PROD",
    ) -> Dict[str, Any]:
        """
        Check alpha correlation against production alphas.
        
        Args:
            alpha_id: The alpha ID to check
            check_type: Type of correlation check (PROD, SELF, etc.)
            
        Returns:
            Correlation check results
        """
        ...
    
    async def get_user_alphas(
        self,
        limit: int = 100,
        offset: int = 0,
        stage: Optional[str] = None,
        search: Optional[str] = None,
        start_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get user's alphas with pagination.
        
        Args:
            limit: Maximum number of results
            offset: Pagination offset
            stage: Filter by alpha stage
            search: Search query
            start_date: Filter by creation date
            
        Returns:
            Dict with 'results' list and 'count' total
        """
        ...
