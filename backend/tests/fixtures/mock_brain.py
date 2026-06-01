"""
Mock Brain Adapter - Test implementation of BrainProtocol

Provides configurable mock responses for testing without
hitting the real BRAIN API.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field


@dataclass
class MockSimulationConfig:
    """Configuration for mock simulation responses."""
    default_sharpe: float = 1.2
    default_fitness: float = 0.75
    default_turnover: float = 0.25
    success_rate: float = 0.8  # Probability of simulation succeeding
    delay_ms: int = 0  # Simulated delay


class MockBrainAdapter:
    """
    Mock implementation of BrainProtocol for testing.
    
    Provides configurable responses for all BRAIN API methods
    without making real network calls.
    
    Usage:
        mock = MockBrainAdapter()
        mock.set_simulation_response(sharpe=1.5, fitness=0.9)
        result = await mock.simulate_alpha("rank(close)")
    """
    
    def __init__(self, config: MockSimulationConfig = None):
        self.config = config or MockSimulationConfig()
        self._simulation_queue: List[Dict] = []
        self._datasets: List[Dict] = []
        self._datafields: Dict[str, List[Dict]] = {}
        self._operators: List[Dict] = []
        self._call_history: List[Dict] = []
        self._setup_defaults()
    
    def _setup_defaults(self):
        """
        Setup default mock data matching real BRAIN API structure.
        """
        # Datasets match real API structure
        self._datasets = [
            {"id": "fundamental6", "name": "Company Fundamental Data for Equity", "category": "fundamental"},
            {"id": "technical5", "name": "Technical 5", "category": "technical"},
            {"id": "price_volume4", "name": "Price Volume 4", "category": "price"},
        ]
        
        # Datafields match real API structure with nested objects
        self._datafields["fundamental6"] = [
            {
                "id": "close",
                "description": "Close Price",
                "dataset": {"id": "fundamental6", "name": "Company Fundamental Data for Equity"},
                "category": {"id": "fundamental", "name": "Fundamental"},
                "subcategory": {"id": "fundamental-price", "name": "Price Data"},
                "region": "USA",
                "delay": 1,
                "universe": "TOP3000",
                "type": "MATRIX",
                "dateCoverage": 1.0,
                "coverage": 0.99,
                "userCount": 50000,
                "alphaCount": 500000,
                "pyramidMultiplier": 1.2,
                "themes": [],
            },
            {
                "id": "open",
                "description": "Open Price",
                "dataset": {"id": "fundamental6", "name": "Company Fundamental Data for Equity"},
                "category": {"id": "fundamental", "name": "Fundamental"},
                "subcategory": {"id": "fundamental-price", "name": "Price Data"},
                "region": "USA",
                "delay": 1,
                "universe": "TOP3000",
                "type": "MATRIX",
                "dateCoverage": 1.0,
                "coverage": 0.99,
                "userCount": 30000,
                "alphaCount": 200000,
                "pyramidMultiplier": 1.2,
                "themes": [],
            },
            {
                "id": "volume",
                "description": "Trading Volume",
                "dataset": {"id": "fundamental6", "name": "Company Fundamental Data for Equity"},
                "category": {"id": "fundamental", "name": "Fundamental"},
                "subcategory": {"id": "fundamental-volume", "name": "Volume Data"},
                "region": "USA",
                "delay": 1,
                "universe": "TOP3000",
                "type": "MATRIX",
                "dateCoverage": 1.0,
                "coverage": 0.99,
                "userCount": 40000,
                "alphaCount": 300000,
                "pyramidMultiplier": 1.2,
                "themes": [],
            },
        ]
        
        # Operators match real API structure
        self._operators = [
            {
                "name": "rank",
                "category": "Cross-Sectional",
                "scope": ["COMBO", "REGULAR", "SELECTION"],
                "definition": "rank(x)",
                "description": "Cross-sectional rank",
                "documentation": "/operators/rank",
                "level": "ALL",
            },
            {
                "name": "ts_rank",
                "category": "Time Series",
                "scope": ["COMBO", "REGULAR", "SELECTION"],
                "definition": "ts_rank(x, d)",
                "description": "Time series rank over d days",
                "documentation": "/operators/ts_rank",
                "level": "ALL",
            },
            {
                "name": "ts_mean",
                "category": "Time Series",
                "scope": ["COMBO", "REGULAR", "SELECTION"],
                "definition": "ts_mean(x, d)",
                "description": "Time series mean over d days",
                "documentation": "/operators/ts_mean",
                "level": "ALL",
            },
            {
                "name": "ts_std_dev",
                "category": "Time Series",
                "scope": ["COMBO", "REGULAR", "SELECTION"],
                "definition": "ts_std_dev(x, d)",
                "description": "Time series standard deviation over d days",
                "documentation": "/operators/ts_std_dev",
                "level": "ALL",
            },
        ]
    
    def reset(self):
        """Reset mock state."""
        self._simulation_queue = []
        self._call_history = []
    
    # =========================================================================
    # Mock Configuration Methods
    # =========================================================================
    
    def set_simulation_response(
        self,
        sharpe: float = None,
        fitness: float = None,
        turnover: float = None,
        success: bool = True,
        error: str = None,
    ):
        """
        Queue a specific simulation response.
        
        Args:
            sharpe: Sharpe ratio to return
            fitness: Fitness score to return
            turnover: Turnover to return
            success: Whether simulation succeeds
            error: Error message if not successful
        """
        response = {
            "success": success,
            "alpha_id": f"mock-alpha-{len(self._simulation_queue)}",
            "metrics": {
                "sharpe": sharpe if sharpe is not None else self.config.default_sharpe,
                "fitness": fitness if fitness is not None else self.config.default_fitness,
                "turnover": turnover if turnover is not None else self.config.default_turnover,
            },
            "error": error,
        }
        self._simulation_queue.append(response)
    
    def set_datasets(self, datasets: List[Dict]):
        """Set datasets to return from get_datasets."""
        self._datasets = datasets
    
    def set_datafields(self, dataset_id: str, fields: List[Dict]):
        """Set fields to return for a dataset."""
        self._datafields[dataset_id] = fields
    
    def set_operators(self, operators: List[Any]):
        """Set operators to return."""
        self._operators = operators
    
    def get_call_history(self) -> List[Dict]:
        """Get history of API calls made."""
        return self._call_history
    
    # =========================================================================
    # BrainProtocol Implementation
    # =========================================================================
    
    async def ensure_session(self) -> None:
        """Mock session - always succeeds."""
        self._call_history.append({"method": "ensure_session"})
    
    async def authenticate(self) -> bool:
        """Mock authentication - always succeeds."""
        self._call_history.append({"method": "authenticate"})
        return True
    
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
        Mock alpha simulation.
        
        Returns queued response if available, otherwise generates
        a default response matching real BRAIN API structure.
        """
        self._call_history.append({
            "method": "simulate_alpha",
            "expression": expression,
            "region": region,
            "universe": universe,
        })
        
        # Use queued response if available
        if self._simulation_queue:
            response = self._simulation_queue.pop(0)
        else:
            # Generate default response matching real API structure
            import random
            success = random.random() < self.config.success_rate
            
            if success:
                sharpe = self.config.default_sharpe + random.uniform(-0.3, 0.3)
                fitness = self.config.default_fitness + random.uniform(-0.1, 0.1)
                turnover = self.config.default_turnover + random.uniform(-0.05, 0.05)
                returns = 0.15 + random.uniform(-0.05, 0.05)
                drawdown = 0.1 + random.uniform(0, 0.05)
                
                # Generate mock checks matching real API
                checks = [
                    {"name": "LOW_SHARPE", "result": "PASS" if sharpe >= 1.5 else "FAIL", "limit": 1.58, "value": sharpe},
                    {"name": "LOW_FITNESS", "result": "PASS" if fitness >= 0.6 else "FAIL", "limit": 1.0, "value": fitness},
                    {"name": "HIGH_TURNOVER", "result": "PASS" if turnover <= 0.7 else "FAIL", "limit": 0.7, "value": turnover},
                    {"name": "LOW_TURNOVER", "result": "PASS" if turnover >= 0.01 else "FAIL", "limit": 0.01, "value": turnover},
                    {"name": "SELF_CORRELATION", "result": "PENDING"},
                    {"name": "PROD_CORRELATION", "result": "PENDING"},
                ]
                
                failed_checks = [c["name"] for c in checks if c.get("result") == "FAIL"]
                pending_checks = [c["name"] for c in checks if c.get("result") == "PENDING"]
                passed_checks = [c["name"] for c in checks if c.get("result") == "PASS"]
                
                # Build nested stats structure matching real API
                is_stats = {
                    "pnl": random.randint(100000, 10000000),
                    "bookSize": 20000000,
                    "longCount": random.randint(50, 200),
                    "shortCount": random.randint(50, 200),
                    "turnover": turnover,
                    "returns": returns,
                    "drawdown": drawdown,
                    "margin": random.uniform(0.0001, 0.001),
                    "sharpe": sharpe,
                    "fitness": fitness,
                    "startDate": "2014-01-01",
                    "investabilityConstrained": {
                        "sharpe": sharpe * 0.8,
                        "fitness": fitness * 0.9,
                        "turnover": turnover * 0.7,
                    },
                    "riskNeutralized": {
                        "sharpe": sharpe * 0.5,
                        "fitness": fitness * 0.6,
                    },
                    "checks": checks,
                }
                
                train_stats = {
                    "sharpe": sharpe * 1.1,
                    "fitness": fitness * 1.05,
                    "turnover": turnover,
                    "returns": returns * 1.1,
                    "drawdown": drawdown,
                }
                
                test_stats = {
                    "sharpe": sharpe * 0.9,
                    "fitness": fitness * 0.95,
                    "turnover": turnover,
                    "returns": returns * 0.9,
                    "drawdown": drawdown * 1.2,
                }
                
                response = {
                    "success": True,
                    "alpha_id": f"mock-{hash(expression) % 10000}",
                    "expression": expression,
                    "stage": "IS",
                    "status": "UNSUBMITTED",
                    "type": "REGULAR",
                    "dateCreated": "2026-01-29T00:00:00Z",
                    "settings": {
                        "instrumentType": "EQUITY",
                        "region": region,
                        "universe": universe,
                        "delay": delay,
                        "decay": decay,
                        "neutralization": neutralization,
                        "truncation": truncation,
                        "testPeriod": test_period,
                    },
                    "metrics": {
                        "sharpe": sharpe,
                        "fitness": fitness,
                        "turnover": turnover,
                        "returns": returns,
                        "drawdown": drawdown,
                        "pnl": is_stats["pnl"],
                        "margin": is_stats["margin"],
                        "train_sharpe": train_stats["sharpe"],
                        "train_fitness": train_stats["fitness"],
                        "test_sharpe": test_stats["sharpe"],
                        "test_fitness": test_stats["fitness"],
                        "investabilityConstrained": is_stats["investabilityConstrained"],
                        "riskNeutralized": is_stats["riskNeutralized"],
                    },
                    "checks": checks,
                    "failed_checks": failed_checks,
                    "pending_checks": pending_checks,
                    "passed_checks": passed_checks,
                    "can_submit": len(failed_checks) == 0 and len(pending_checks) == 0,
                    "is": is_stats,
                    "train": train_stats,
                    "test": test_stats,
                    "os": None,
                }
            else:
                response = {
                    "success": False,
                    "error": "Mock simulation error",
                }
        
        return response
    
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
        Mock batch simulation.
        
        Simulates each expression individually.
        """
        self._call_history.append({
            "method": "simulate_batch",
            "count": len(expressions),
            "region": region,
        })
        
        results = []
        for expr in expressions:
            result = await self.simulate_alpha(
                expression=expr,
                region=region,
                universe=universe,
                delay=delay,
                decay=decay,
                neutralization=neutralization,
                truncation=truncation,
                test_period=test_period,
            )
            results.append(result)
        
        return results
    
    async def get_datasets(
        self,
        region: str = "USA",
        delay: int = 1,
        universe: str = "TOP3000",
    ) -> List[Dict[str, Any]]:
        """Return mock datasets."""
        self._call_history.append({
            "method": "get_datasets",
            "region": region,
        })
        return self._datasets
    
    async def get_datafields(
        self,
        dataset_id: str,
        region: str = "USA",
        delay: int = 1,
        universe: str = "TOP3000",
    ) -> List[Dict[str, Any]]:
        """Return mock datafields."""
        self._call_history.append({
            "method": "get_datafields",
            "dataset_id": dataset_id,
        })
        return self._datafields.get(dataset_id, [])
    
    async def get_operators(self, detailed: bool = False) -> List[Any]:
        """Return mock operators."""
        self._call_history.append({
            "method": "get_operators",
            "detailed": detailed,
        })
        if detailed:
            return self._operators
        return [op["name"] if isinstance(op, dict) else op for op in self._operators]
    
    async def get_alpha_pnl(self, alpha_id: str) -> Dict[str, Any]:
        """Return mock PnL data."""
        self._call_history.append({
            "method": "get_alpha_pnl",
            "alpha_id": alpha_id,
        })
        return {"dates": [], "pnl": [], "cumulative": []}
    
    async def check_correlation(
        self,
        alpha_id: str,
        check_type: str = "PROD",
    ) -> Dict[str, Any]:
        """Return mock correlation data."""
        self._call_history.append({
            "method": "check_correlation",
            "alpha_id": alpha_id,
            "check_type": check_type,
        })
        return {"correlations": [], "max_correlation": 0.3}
    
    async def get_user_alphas(
        self,
        limit: int = 100,
        offset: int = 0,
        stage: Optional[str] = None,
        search: Optional[str] = None,
        start_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return mock user alphas."""
        self._call_history.append({
            "method": "get_user_alphas",
            "limit": limit,
            "offset": offset,
        })
        return {"results": [], "count": 0}
    
    # =========================================================================
    # Context Manager Support
    # =========================================================================
    
    async def __aenter__(self):
        await self.ensure_session()
        return self
    
    async def __aexit__(self, *args):
        pass
