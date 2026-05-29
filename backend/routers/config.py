"""
Config Router - System configuration management

Uses ConfigService for business logic and CredentialsService for credentials.

Includes:
- Quality thresholds
- Operator preferences
- Credentials management (Brain, LLM API)
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field

from backend.database import get_db
from backend.services.config_service import (
    ConfigService,
    ThresholdsConfig as ThresholdsConfigData,
    DiversityConfig as DiversityConfigData,
)
from backend.services.credentials_service import (
    CredentialsService,
    CredentialKey,
    get_credentials_service,
)

router = APIRouter(
    prefix="/config",
    tags=["config"],
    responses={404: {"description": "Not found"}},
)


# =============================================================================
# DEPENDENCY INJECTION
# =============================================================================

def get_config_service(db: AsyncSession = Depends(get_db)) -> ConfigService:
    """Get ConfigService instance with injected dependencies."""
    return ConfigService(db)


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================

class ThresholdsConfig(BaseModel):
    sharpe_min: float = 1.58
    turnover_max: float = 0.3
    fitness_min: float = 1.0
    returns_min: float = 0.0
    max_dd_max: float = 0.3


class DiversityConfig(BaseModel):
    max_correlation: float = 0.7


class FullConfig(BaseModel):
    quality_thresholds: Optional[ThresholdsConfig] = None
    diversity_thresholds: Optional[DiversityConfig] = None
    daily_budget: Optional[dict] = None


class BrainCredentialsRequest(BaseModel):
    """Request model for Brain platform credentials."""
    email: str = Field(..., description="Brain platform email")
    password: str = Field(..., description="Brain platform password")


class LLMCredentialsRequest(BaseModel):
    """Request model for LLM API credentials."""
    api_key: str = Field(..., description="API key (e.g., OpenAI, DeepSeek)")
    base_url: str = Field(
        default="https://api.deepseek.com",
        description="API base URL"
    )
    model: str = Field(
        default="deepseek-chat",
        description="Model name"
    )


class CredentialStatusResponse(BaseModel):
    """Response model for credential status."""
    key: str
    masked: str
    is_set: bool
    source: Optional[str] = None
    updated_at: Optional[str] = None


class OperatorPrefResponse(BaseModel):
    operator_name: str
    status: str
    usage_count: int
    success_count: int
    failure_rate: float


# =============================================================================
# CREDENTIALS MANAGEMENT (Must be before /{key} route)
# =============================================================================

@router.get("/credentials")
async def get_credentials_status(db: AsyncSession = Depends(get_db)):
    """Get status of all configured credentials (masked values)."""
    service = get_credentials_service(db)
    credentials = await service.get_all_credentials_masked()
    
    return {
        "credentials": credentials,
        "message": "Use POST endpoints to update credentials"
    }


@router.post("/credentials/brain")
async def set_brain_credentials(
    credentials: BrainCredentialsRequest,
    db: AsyncSession = Depends(get_db)
):
    """Set WorldQuant Brain platform credentials."""
    from backend.adapters.brain_adapter import BrainAdapter
    
    service = get_credentials_service(db)
    
    try:
        await service.set_credential(
            CredentialKey.BRAIN_EMAIL,
            credentials.email,
            description="WorldQuant Brain platform email"
        )
        await service.set_credential(
            CredentialKey.BRAIN_PASSWORD,
            credentials.password,
            description="WorldQuant Brain platform password"
        )
        
        # Invalidate cached credentials
        BrainAdapter.invalidate_credentials_cache()
        CredentialsService.invalidate_cache()
        
        return {
            "success": True,
            "message": "Brain credentials saved successfully"
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save credentials: {str(e)}"
        )


@router.post("/credentials/llm")
async def set_llm_credentials(
    credentials: LLMCredentialsRequest,
    db: AsyncSession = Depends(get_db)
):
    """Set LLM API credentials (OpenAI, DeepSeek, etc.)."""
    service = get_credentials_service(db)
    
    try:
        await service.set_credential(
            CredentialKey.OPENAI_API_KEY,
            credentials.api_key,
            description="LLM API key"
        )
        await service.set_credential(
            CredentialKey.OPENAI_BASE_URL,
            credentials.base_url,
            description="LLM API base URL"
        )
        await service.set_credential(
            CredentialKey.OPENAI_MODEL,
            credentials.model,
            description="LLM model name"
        )
        
        # Invalidate credential caches
        CredentialsService.invalidate_cache()
        try:
            from backend.agents.services.llm_service import get_llm_service
            get_llm_service().invalidate_credentials_cache()
        except Exception:
            pass
        
        return {
            "success": True,
            "message": "LLM credentials saved successfully"
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save credentials: {str(e)}"
        )


@router.post("/credentials/brain/test")
async def test_brain_credentials(db: AsyncSession = Depends(get_db)):
    """Test Brain platform credentials by attempting authentication."""
    service = get_credentials_service(db)
    result = await service.test_brain_credentials()
    
    if not result["success"]:
        raise HTTPException(
            status_code=400,
            detail=result.get("error", "Authentication failed")
        )
    
    return result


@router.delete("/credentials/{key}")
async def delete_credential(
    key: str,
    db: AsyncSession = Depends(get_db)
):
    """Delete a specific credential."""
    valid_keys = [
        CredentialKey.BRAIN_EMAIL,
        CredentialKey.BRAIN_PASSWORD,
        CredentialKey.OPENAI_API_KEY,
        CredentialKey.OPENAI_BASE_URL,
        CredentialKey.OPENAI_MODEL,
    ]
    
    if key not in valid_keys:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid credential key. Valid keys: {valid_keys}"
        )
    
    service = get_credentials_service(db)
    deleted = await service.delete_credential(key)
    
    if not deleted:
        raise HTTPException(
            status_code=404,
            detail=f"Credential '{key}' not found"
        )
    
    return {"success": True, "message": f"Credential '{key}' deleted"}


# =============================================================================
# THRESHOLDS & DIVERSITY
# =============================================================================

@router.get("/thresholds")
async def get_thresholds(
    service: ConfigService = Depends(get_config_service),
):
    """Get quality thresholds configuration."""
    config = await service.get_thresholds()
    return {
        "sharpe_min": config.sharpe_min,
        "turnover_max": config.turnover_max,
        "fitness_min": config.fitness_min,
        "returns_min": config.returns_min,
        "max_dd_max": config.max_dd_max,
    }


@router.put("/thresholds")
async def update_thresholds(
    thresholds: ThresholdsConfig,
    service: ConfigService = Depends(get_config_service),
):
    """Update quality thresholds."""
    config = ThresholdsConfigData(
        sharpe_min=thresholds.sharpe_min,
        turnover_max=thresholds.turnover_max,
        fitness_min=thresholds.fitness_min,
        returns_min=thresholds.returns_min,
        max_dd_max=thresholds.max_dd_max,
    )
    
    updated = await service.update_thresholds(config)
    
    return {
        "message": "Thresholds updated",
        "thresholds": {
            "sharpe_min": updated.sharpe_min,
            "turnover_max": updated.turnover_max,
            "fitness_min": updated.fitness_min,
            "returns_min": updated.returns_min,
            "max_dd_max": updated.max_dd_max,
        }
    }


@router.put("/diversity")
async def update_diversity(
    diversity: DiversityConfig,
    service: ConfigService = Depends(get_config_service),
):
    """Update diversity thresholds."""
    config = DiversityConfigData(
        max_correlation=diversity.max_correlation,
    )
    
    updated = await service.update_diversity_config(config)
    
    return {
        "message": "Diversity config updated",
        "diversity": {
            "max_correlation": updated.max_correlation,
        }
    }


@router.get("/diversity")
async def get_diversity(
    service: ConfigService = Depends(get_config_service),
):
    """Get diversity thresholds configuration."""
    config = await service.get_diversity_config()
    return {
        "max_correlation": config.max_correlation,
    }


# =============================================================================
# OPERATORS
# =============================================================================

@router.get("/operators", response_model=List[OperatorPrefResponse])
async def get_operator_prefs(
    service: ConfigService = Depends(get_config_service),
):
    """Get all operator preferences."""
    prefs = await service.get_operator_preferences()
    
    return [
        OperatorPrefResponse(
            operator_name=p.operator_name,
            status=p.status,
            usage_count=p.usage_count,
            success_count=p.success_count,
            failure_rate=p.failure_rate,
        )
        for p in prefs
    ]


@router.put("/operators/{operator_name}")
async def update_operator_pref(
    operator_name: str,
    status: str,
    service: ConfigService = Depends(get_config_service),
):
    """Update operator status (ACTIVE, BANNED, DEPRECATED)."""
    try:
        await service.update_operator_status(operator_name, status)
        return {"message": f"Operator {operator_name} set to {status}"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# =============================================================================
# GENERAL CONFIG (Must be last due to {key} path param)
# =============================================================================

@router.get("")
async def get_all_config(
    service: ConfigService = Depends(get_config_service),
):
    """Get all system configuration (excluding credentials)."""
    return await service.get_all_config()
