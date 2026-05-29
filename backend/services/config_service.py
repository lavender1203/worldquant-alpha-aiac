"""
Config Service - Business logic for system configuration management

Provides methods for:
- Quality thresholds management
- Diversity thresholds management
- Operator preferences management
- General system config
"""

import json
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update

from backend.services.base import BaseService
from backend.models import SystemConfig, OperatorPreference, Operator

logger = logging.getLogger("services.config")


@dataclass
class ThresholdsConfig:
    """Quality thresholds configuration."""
    sharpe_min: float = 1.58
    turnover_max: float = 0.3
    fitness_min: float = 1.0
    returns_min: float = 0.0
    max_dd_max: float = 0.3


@dataclass
class DiversityConfig:
    """Diversity thresholds configuration."""
    max_correlation: float = 0.7


@dataclass
class OperatorPrefInfo:
    """Operator preference information."""
    operator_name: str
    status: str
    usage_count: int
    success_count: int
    failure_rate: float


class ConfigService(BaseService):
    """
    Service for system configuration management.
    
    Provides a clean interface for config management,
    abstracting database operations from routers.
    """
    
    # =========================================================================
    # Thresholds Operations
    # =========================================================================
    
    async def get_thresholds(self) -> ThresholdsConfig:
        """
        Get quality thresholds configuration.
        
        Returns:
            ThresholdsConfig with current values or defaults
        """
        query = select(SystemConfig).where(
            SystemConfig.config_key == "quality_thresholds"
        )
        result = await self.db.execute(query)
        config = result.scalar_one_or_none()
        
        if config and config.config_value:
            try:
                data = json.loads(config.config_value)
                return ThresholdsConfig(**data)
            except Exception:
                pass
        
        return ThresholdsConfig()
    
    async def update_thresholds(
        self,
        thresholds: ThresholdsConfig,
    ) -> ThresholdsConfig:
        """
        Update quality thresholds.
        
        Args:
            thresholds: New threshold values
            
        Returns:
            Updated ThresholdsConfig
        """
        value = json.dumps({
            "sharpe_min": thresholds.sharpe_min,
            "turnover_max": thresholds.turnover_max,
            "fitness_min": thresholds.fitness_min,
            "returns_min": thresholds.returns_min,
            "max_dd_max": thresholds.max_dd_max,
        })
        
        await self._upsert_config(
            key="quality_thresholds",
            value=value,
            config_type="json",
            description="Alpha quality thresholds",
        )
        
        return thresholds
    
    async def get_diversity_config(self) -> DiversityConfig:
        """
        Get diversity thresholds configuration.
        
        Returns:
            DiversityConfig with current values or defaults
        """
        query = select(SystemConfig).where(
            SystemConfig.config_key == "diversity_thresholds"
        )
        result = await self.db.execute(query)
        config = result.scalar_one_or_none()
        
        if config and config.config_value:
            try:
                data = json.loads(config.config_value)
                return DiversityConfig(**data)
            except Exception:
                pass
        
        return DiversityConfig()
    
    async def update_diversity_config(
        self,
        diversity: DiversityConfig,
    ) -> DiversityConfig:
        """
        Update diversity thresholds.
        
        Args:
            diversity: New diversity values
            
        Returns:
            Updated DiversityConfig
        """
        value = json.dumps({
            "max_correlation": diversity.max_correlation,
        })
        
        await self._upsert_config(
            key="diversity_thresholds",
            value=value,
            config_type="json",
            description="Alpha diversity thresholds",
        )
        
        return diversity
    
    # =========================================================================
    # Operator Preferences Operations
    # =========================================================================
    
    async def get_operator_preferences(self) -> List[OperatorPrefInfo]:
        """
        Get all operator preferences.
        
        Returns:
            List of OperatorPrefInfo
        """
        pref_query = select(OperatorPreference).order_by(
            OperatorPreference.usage_count.desc()
        )
        pref_result = await self.db.execute(pref_query)
        prefs = pref_result.scalars().all()
        
        items = [
            OperatorPrefInfo(
                operator_name=op.operator_name,
                status=op.status,
                usage_count=op.usage_count,
                success_count=op.success_count,
                failure_rate=op.failure_rate,
            )
            for op in prefs
        ]

        seen = {item.operator_name for item in items}
        operator_query = select(Operator).order_by(Operator.name.asc())
        operator_result = await self.db.execute(operator_query)
        for operator in operator_result.scalars().all():
            if operator.name in seen:
                continue
            items.append(
                OperatorPrefInfo(
                    operator_name=operator.name,
                    status="ACTIVE" if operator.is_active else "BANNED",
                    usage_count=0,
                    success_count=0,
                    failure_rate=0.0,
                )
            )

        return items
    
    async def update_operator_status(
        self,
        operator_name: str,
        status: str,
    ) -> bool:
        """
        Update operator status.
        
        Args:
            operator_name: Operator name
            status: New status (ACTIVE, BANNED, DEPRECATED)
            
        Returns:
            True if updated
            
        Raises:
            ValueError if invalid status
        """
        valid_statuses = ["ACTIVE", "BANNED", "DEPRECATED"]
        if status not in valid_statuses:
            raise ValueError(f"Invalid status. Valid values: {valid_statuses}")
        
        query = select(OperatorPreference).where(
            OperatorPreference.operator_name == operator_name
        )
        result = await self.db.execute(query)
        pref = result.scalar_one_or_none()

        if pref:
            pref.status = status
        else:
            self.db.add(OperatorPreference(
                operator_name=operator_name,
                status=status,
                usage_count=0,
                success_count=0,
                failure_rate=0.0,
            ))

        await self.db.execute(
            update(Operator)
            .where(Operator.name == operator_name)
            .values(is_active=(status == "ACTIVE"))
        )
        await self.commit()
        
        return True
    
    # =========================================================================
    # General Config Operations
    # =========================================================================
    
    async def get_all_config(self) -> Dict[str, Any]:
        """
        Get all system configuration (excluding credentials).
        
        Returns:
            Dict of config key-value pairs
        """
        query = select(SystemConfig).where(
            ~SystemConfig.config_key.like("credential:%")
        )
        result = await self.db.execute(query)
        configs = result.scalars().all()
        
        return {c.config_key: c.config_value for c in configs}
    
    async def get_config(self, key: str) -> Optional[str]:
        """
        Get a specific config value.
        
        Args:
            key: Config key
            
        Returns:
            Config value or None
        """
        query = select(SystemConfig).where(SystemConfig.config_key == key)
        result = await self.db.execute(query)
        config = result.scalar_one_or_none()
        
        return config.config_value if config else None
    
    async def set_config(
        self,
        key: str,
        value: str,
        config_type: str = "string",
        description: Optional[str] = None,
    ) -> bool:
        """
        Set a config value.
        
        Args:
            key: Config key
            value: Config value
            config_type: Value type
            description: Optional description
            
        Returns:
            True if set
        """
        await self._upsert_config(key, value, config_type, description)
        return True
    
    # =========================================================================
    # Helper Methods
    # =========================================================================
    
    async def _upsert_config(
        self,
        key: str,
        value: str,
        config_type: str = "string",
        description: Optional[str] = None,
    ) -> None:
        """
        Insert or update a config entry.
        
        Args:
            key: Config key
            value: Config value
            config_type: Value type
            description: Optional description
        """
        query = select(SystemConfig).where(SystemConfig.config_key == key)
        result = await self.db.execute(query)
        existing = result.scalar_one_or_none()
        
        if existing:
            existing.config_value = value
            if description:
                existing.description = description
        else:
            new_config = SystemConfig(
                config_key=key,
                config_value=value,
                config_type=config_type,
                description=description,
            )
            self.db.add(new_config)
        
        await self.commit()
