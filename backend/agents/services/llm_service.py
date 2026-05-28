"""
LLM Service - Unified LLM calling interface with logging and retries

Implements LLMProtocol for dependency injection and testability.
High cohesion: All LLM-related logic in one place.
"""

import asyncio
import json
import os
import time
from typing import Dict, List, Optional, Any, Type, Tuple
from pydantic import BaseModel
import openai
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from loguru import logger

from backend.config import settings
from backend.protocols.llm_protocol import LLMProtocol, LLMResponse as LLMResponseProtocol


class LLMResponse(BaseModel):
    """Standard LLM response wrapper."""
    content: str
    parsed: Optional[Dict] = None
    model: str
    tokens_used: int = 0
    latency_ms: int = 0
    success: bool = True
    error: Optional[str] = None
    
    def to_protocol_response(self) -> LLMResponseProtocol:
        """Convert to protocol response type."""
        return LLMResponseProtocol(
            content=self.content,
            parsed=self.parsed,
            model=self.model,
            tokens_used=self.tokens_used,
            latency_ms=self.latency_ms,
            success=self.success,
            error=self.error,
        )


class LLMService:
    """
    Unified LLM Service implementing LLMProtocol.
    
    Features:
    - Automatic retries with exponential backoff
    - JSON cleaning (markdown removal)
    - Token tracking
    - Structured logging
    - Credential caching with invalidation support
    
    This class implements the LLMProtocol interface, allowing for
    easy mocking in tests and dependency injection.
    """
    
    def __init__(
        self,
        api_key: str = None,
        base_url: str = None,
        model: str = None
    ):
        self.api_key = api_key or settings.OPENAI_API_KEY
        self.base_url = base_url or settings.OPENAI_BASE_URL
        self.model = model or getattr(settings, 'OPENAI_MODEL', 'deepseek-chat')

        self._credentials_lock = asyncio.Lock()
        self._credentials_loaded = False
        
        self.client = openai.AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=getattr(settings, "LLM_TIMEOUT_SECONDS", 60),
            max_retries=0,
        )
        
        logger.info(f"[LLMService] Initialized | model={self.model} base_url={self.base_url}")

    async def _ensure_credentials_loaded(self):
        if self._credentials_loaded:
            return

        async with self._credentials_lock:
            if self._credentials_loaded:
                return

            try:
                from backend.database import AsyncSessionLocal
                from backend.services.credentials_service import CredentialsService, CredentialKey

                env_api_key = os.getenv("OPENAI_API_KEY")
                env_base_url = os.getenv("OPENAI_BASE_URL")
                env_model = os.getenv("OPENAI_MODEL")

                async with AsyncSessionLocal() as db:
                    service = CredentialsService(db)
                    db_api_key = None if env_api_key else await service.get_credential(CredentialKey.OPENAI_API_KEY)
                    db_base_url = None if env_base_url else await service.get_credential(CredentialKey.OPENAI_BASE_URL)
                    db_model = None if env_model else await service.get_credential(CredentialKey.OPENAI_MODEL)

                if env_api_key or db_api_key:
                    self.api_key = env_api_key or db_api_key
                if env_base_url or db_base_url:
                    self.base_url = env_base_url or db_base_url
                if env_model or db_model:
                    self.model = env_model or db_model

                self.client = openai.AsyncOpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url,
                    timeout=getattr(settings, "LLM_TIMEOUT_SECONDS", 60),
                    max_retries=0,
                )
            except Exception as e:
                logger.warning(f"[LLMService] Failed to load DB credentials, using settings/env | error={e}")
            finally:
                self._credentials_loaded = True

    def invalidate_credentials_cache(self):
        self._credentials_loaded = False
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((openai.APIConnectionError, openai.RateLimitError))
    )
    async def call(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        json_mode: bool = True,
        max_tokens: int = 4096
    ) -> LLMResponse:
        """
        Make an LLM call with automatic retries and logging.
        
        Args:
            system_prompt: System message
            user_prompt: User message
            temperature: Sampling temperature
            json_mode: Whether to request JSON output
            max_tokens: Maximum response tokens
            
        Returns:
            LLMResponse with content and metadata
        """
        start_time = time.time()
        call_id = f"{int(start_time * 1000) % 100000}"
        
        logger.debug(f"[LLMService] Call started | id={call_id} json_mode={json_mode}")
        
        try:
            await self._ensure_credentials_loaded()
            
            # Kimi k2.6 and some other o1-like models only support temperature=1
            actual_temp = temperature
            if "kimi-k2.6" in self.model:
                actual_temp = 1.0
            disable_thinking = self._should_disable_thinking()
            if disable_thinking and "kimi-k2.6" in (self.model or "").lower():
                actual_temp = 0.6
                
            request_kwargs = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": actual_temp,
                "max_tokens": max_tokens,
                "response_format": {"type": "json_object"} if json_mode else None,
            }

            if disable_thinking:
                request_kwargs["extra_body"] = {"thinking": {"type": "disabled"}}

            response = await self.client.chat.completions.create(**request_kwargs)
            
            # Defensive: handle empty/malformed responses
            choices = getattr(response, "choices", None)
            if not response or not choices:
                status = getattr(response, "status", None)
                msg = getattr(response, "msg", None)
                extra = f" | status={status} msg={msg}" if status or msg else ""
                raise ValueError(f"Empty response from LLM API{extra}")

            if len(choices) == 0:
                raise ValueError("Empty choices from LLM API")
            
            message = response.choices[0].message
            if not message:
                raise ValueError("No message in LLM response")
                
            content = message.content or ""
            if json_mode and not content.strip():
                finish_reason = getattr(response.choices[0], "finish_reason", None)
                reasoning_content = getattr(message, "reasoning_content", None)
                extra = f"finish_reason={finish_reason}" if finish_reason else ""
                if reasoning_content:
                    extra = (extra + " | reasoning_content_present=True").strip()
                raise ValueError(f"Empty content in LLM response ({extra})")
            tokens_used = response.usage.total_tokens if response.usage else 0
            latency_ms = int((time.time() - start_time) * 1000)
            
            # Parse JSON if requested
            parsed = None
            if json_mode:
                try:
                    cleaned = self._clean_json(content)
                    parsed = json.loads(cleaned)
                except json.JSONDecodeError as e:
                    logger.warning(f"[LLMService] JSON parse failed | id={call_id} error={e}")
            
            logger.info(
                f"[LLMService] Call success | id={call_id} "
                f"tokens={tokens_used} latency={latency_ms}ms"
            )
            
            return LLMResponse(
                content=content,
                parsed=parsed,
                model=self.model,
                tokens_used=tokens_used,
                latency_ms=latency_ms,
                success=True
            )
            
        except Exception as e:
            latency_ms = int((time.time() - start_time) * 1000)
            logger.error(f"[LLMService] Call failed | id={call_id} error={e}")
            
            return LLMResponse(
                content="",
                model=self.model,
                latency_ms=latency_ms,
                success=False,
                error=str(e)
            )
    
    async def call_with_schema(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: Type[BaseModel],
        temperature: float = 0.7
    ) -> tuple[Optional[BaseModel], LLMResponse]:
        """
        Call LLM and validate response against a Pydantic schema.
        
        Args:
            system_prompt: System message
            user_prompt: User message
            schema: Pydantic model class to validate against
            temperature: Sampling temperature
            
        Returns:
            Tuple of (parsed model or None, raw response)
        """
        response = await self.call(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            json_mode=True
        )
        
        if not response.success or not response.parsed:
            return None, response
        
        try:
            validated = schema.model_validate(response.parsed)
            return validated, response
        except Exception as e:
            logger.warning(f"[LLMService] Schema validation failed | error={e}")
            return None, response
    
    def _clean_json(self, content: str) -> str:
        """Remove markdown code blocks from JSON response."""
        content = content.strip()
        
        # Remove leading markdown
        if content.startswith('```json'):
            content = content[7:]
        elif content.startswith('```'):
            content = content[3:]
        
        # Remove trailing markdown
        if content.endswith('```'):
            content = content[:-3]
        
        return content.strip()

    def _should_disable_thinking(self) -> bool:
        """Disable reasoning mode for JSON workflow calls on supported providers."""
        if not getattr(settings, "LLM_DISABLE_THINKING", True):
            return False

        model = (self.model or "").lower()
        base_url = (self.base_url or "").lower()
        is_kimi = "moonshot" in base_url and ("kimi-k2.5" in model or "kimi-k2.6" in model)
        is_deepseek_v4 = "deepseek" in base_url and model in {"deepseek-v4-flash", "deepseek-v4-pro"}
        return is_kimi or is_deepseek_v4


# Singleton instance for reuse
_llm_service: Optional[LLMService] = None


def get_llm_service() -> LLMService:
    """Get or create singleton LLM service."""
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service
