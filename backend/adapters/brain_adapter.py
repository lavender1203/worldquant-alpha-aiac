"""
BRAIN Adapter - WorldQuant BRAIN Platform API Integration

Implements BrainProtocol for dependency injection and testability.

Refactored based on ace_lib.py best practices:
- Singleton Session (httpx.AsyncClient)
- Active Token Expiry Checking
- Basic Authentication
- Retry-After Handling
"""

import os
import asyncio
import json
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import httpx
import redis.asyncio as redis
from tenacity import retry, stop_after_attempt, wait_exponential
from loguru import logger
from sqlalchemy import select
import logging

# Suppress httpx interaction logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

from backend.config import settings
from backend.database import AsyncSessionLocal
from backend.models import BrainAuthToken

# Import protocol for type checking (Protocol is runtime_checkable)
from backend.protocols.brain_protocol import BrainProtocol

# Singleton Client Storage (Loop-aware)
_GLOBAL_CLIENT: Optional[httpx.AsyncClient] = None
_GLOBAL_CLIENT_LOOP: Optional[asyncio.AbstractEventLoop] = None


def _format_exception(exc: Exception) -> str:
    message = str(exc)
    if message:
        return f"{type(exc).__name__}: {message}"
    return f"{type(exc).__name__}: {exc!r}"

class BrainAdapter:
    """
    Adapter for WorldQuant BRAIN platform.
    Uses a singleton AsyncClient for persistent session management within the same event loop.
    
    Credentials priority:
    1. Constructor arguments (explicit)
    2. Database configuration (via CredentialsService)
    3. Environment variables (fallback)
    """
    
    BASE_URL = "https://api.worldquantbrain.com"
    SESSION_BUFFER_SECONDS = 300  # Re-auth if expiring in < 5 mins
    REDIS_SESSION_KEY = "brain_session:cookies"
    
    # Class-level cached credentials (to avoid DB queries on every request)
    _cached_email: Optional[str] = None
    _cached_password: Optional[str] = None
    _credentials_loaded: bool = False
    
    def __init__(self, email: str = None, password: str = None):
        # Store explicit credentials if provided
        self._explicit_email = email
        self._explicit_password = password
        
        # Initialize with explicit or env fallback (DB credentials loaded async)
        self.email = email or settings.BRAIN_EMAIL
        self.password = password or settings.BRAIN_PASSWORD
        self.session_token = None
    
    async def _load_credentials_from_db(self) -> bool:
        """
        Load credentials from database if not already loaded.
        Returns True if credentials were loaded/updated.
        """
        # Skip if explicit credentials were provided in constructor
        if self._explicit_email and self._explicit_password:
            return False
        
        # Skip if already loaded
        if BrainAdapter._credentials_loaded:
            if BrainAdapter._cached_email:
                self.email = BrainAdapter._cached_email
            if BrainAdapter._cached_password:
                self.password = BrainAdapter._cached_password
            return bool(BrainAdapter._cached_email)
        
        try:
            from backend.services.credentials_service import (
                CredentialsService, 
                CredentialKey
            )
            
            async with AsyncSessionLocal() as db:
                service = CredentialsService(db)
                
                # Load email
                db_email = await service.get_credential(
                    CredentialKey.BRAIN_EMAIL,
                    fallback_env="BRAIN_EMAIL"
                )
                if db_email:
                    BrainAdapter._cached_email = db_email
                    self.email = db_email
                
                # Load password
                db_password = await service.get_credential(
                    CredentialKey.BRAIN_PASSWORD,
                    fallback_env="BRAIN_PASSWORD"
                )
                if db_password:
                    BrainAdapter._cached_password = db_password
                    self.password = db_password
                
                BrainAdapter._credentials_loaded = True
                
                if db_email or db_password:
                    logger.info("Loaded Brain credentials from database")
                    return True
                
        except Exception as e:
            logger.warning(f"Failed to load credentials from DB: {e}")
        
        return False
    
    @classmethod
    def invalidate_credentials_cache(cls):
        """Invalidate cached credentials (call after updating credentials)."""
        cls._cached_email = None
        cls._cached_password = None
        cls._credentials_loaded = False
        logger.info("Brain credentials cache invalidated")
    
    @classmethod
    async def get_client(cls) -> httpx.AsyncClient:
        """Get or create the global singleton client for the current event loop."""
        global _GLOBAL_CLIENT, _GLOBAL_CLIENT_LOOP
        
        current_loop = asyncio.get_running_loop()
        
        # If client exists but loop doesn't match (or loop closed), reset it
        if _GLOBAL_CLIENT:
            if _GLOBAL_CLIENT.is_closed or _GLOBAL_CLIENT_LOOP != current_loop:
                logger.debug("Event loop changed or client closed, resetting BrainAdapter client")
                # Try to close old one if loop still open (unlikely if loop changed) 
                # but we can't await on old loop easily. Just drop ref.
                _GLOBAL_CLIENT = None
                _GLOBAL_CLIENT_LOOP = None

        if _GLOBAL_CLIENT is None:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Origin": "https://platform.worldquantbrain.com",
                "Referer": "https://platform.worldquantbrain.com/",
                "Accept": "application/json;version=2.0"
            }
            _GLOBAL_CLIENT = httpx.AsyncClient(
                timeout=60.0, 
                headers=headers,
                follow_redirects=True
            )
            _GLOBAL_CLIENT_LOOP = current_loop
            
        return _GLOBAL_CLIENT

    async def __aenter__(self):
        self.client = await self.get_client()
        await self.ensure_session()
        return self

    async def __aexit__(self, *args):
        # Do not close the global client here; it persists.
        pass
    
    @classmethod
    async def close(cls):
        """Explicitly close the global client (app shutdown)."""
        global _GLOBAL_CLIENT
        if _GLOBAL_CLIENT:
            await _GLOBAL_CLIENT.aclose()
            _GLOBAL_CLIENT = None

    async def _get_redis(self):
        """Get redis connection"""
        return redis.from_url(settings.REDIS_URL, decode_responses=True)

    async def _load_session_from_redis(self) -> bool:
        """Load cookies from Redis if they exist."""
        try:
            r = await self._get_redis()
            cookies_json = await r.get(self.REDIS_SESSION_KEY)
            await r.aclose()
            
            if cookies_json:
                cookies = json.loads(cookies_json)
                self.client.cookies.update(cookies)
                logger.debug("Loaded session cookies from Redis")
                # When loaded from Redis, we trust it aligns with expiry.
                return True
            return False
        except Exception as e:
            logger.warning(f"Failed to load session from Redis: {e}")
            return False

    async def _save_session_to_redis(self, expiry_seconds: int):
        """Save current cookies to Redis with TTL."""
        try:
            cookies = dict(self.client.cookies)
            if not cookies:
                return
                
            r = await self._get_redis()
            # Set TTL slightly less than actual expiry to be safe (e.g. 5 min buffer logic already in caller or here)
            # If expiry_seconds is "seconds remaining", we use it as TTL directly.
            # If it's a timestamp, we calculate diff? 
            # Brain API returns "expiry": 14400 (seconds remaining). So use directly.
            ttl = max(60, int(expiry_seconds) - 60) # Reduce by 1 min to be safe
            await r.set(self.REDIS_SESSION_KEY, json.dumps(cookies), ex=ttl)
            await r.aclose()
            logger.debug(f"Saved session to Redis (TTL: {ttl}s)")
        except Exception as e:
            logger.error(f"Failed to save session to Redis: {e}")

    async def _invalidate_session_cache(self):
        """Drop stale client/Redis cookies after an authorization failure."""
        try:
            self.client.cookies.clear()
        except Exception:
            pass

        try:
            r = await self._get_redis()
            await r.delete(self.REDIS_SESSION_KEY)
            await r.aclose()
            logger.debug("Invalidated cached Brain session cookies")
        except Exception as e:
            logger.warning(f"Failed to invalidate Redis Brain session: {e}")

    async def ensure_session(self):
        """Ensure valid session exists, refreshing if needed. Prefer Redis cache."""
        # 0. Load credentials from DB if not already loaded
        await self._load_credentials_from_db()
        
        # 1. Try to load from Redis first
        if await self._load_session_from_redis():
            # If loaded from Redis, we assume it is valid for now (TTL handles expiry)
            # We could do a lightweight check, but to save requests, we trust Redis.
            return

        # 2. If no Redis session, check active client state
        if not await self._is_session_valid():
            logger.info("Session invalid or expiring, re-authenticating...")
            await self.authenticate()

    async def _is_session_valid(self) -> bool:
        """
        Check if current session is valid by querying API.
        Reference: ace_lib.py `check_session_timeout`
        """
        try:
            # We need to use the client directly to check
            response = await self.client.get(f"{self.BASE_URL}/authentication")
            
            if response.status_code == 200:
                data = response.json()
                expiry = data.get("token", {}).get("expiry", 0)
                logger.debug(f"Session check: expiry={expiry}, buffer={self.SESSION_BUFFER_SECONDS}")
                # expiry is seconds remaining
                if expiry > self.SESSION_BUFFER_SECONDS:
                    return True
                else:
                    logger.debug(f"Session expiring soon: {expiry}s remaining")
                    return False
            return False
        except Exception:
            return False

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2, min=4, max=60))
    async def authenticate(self) -> bool:
        """
        Authenticate using Basic Auth.
        Reference: ace_lib.py `start_session` uses Basic Auth (via requests.auth).
        """
        try:
            response = await self.client.post(
                f"{self.BASE_URL}/authentication",
                auth=(self.email, self.password)
            )
            
            if response.status_code == 201:
                logger.info("BRAIN authentication successful")
                
                # Save session to Redis
                data = response.json()
                expiry = data.get("token", {}).get("expiry", 3600*4) # Default 4h if missing
                await self._save_session_to_redis(expiry)
                
                return True
            elif response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                if retry_after:
                     logger.warning(f"Rate limited. Sleeping {retry_after}s")
                     await asyncio.sleep(float(retry_after))
                raise Exception("Rate limit exceeded")
            else:
                logger.error(f"Auth failed: {response.status_code} - {response.text}")
                raise Exception(f"Auth failed: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            raise

    # ... Methods (simulate_alpha, get_datasets, etc.) need to use self.client ...
    # I will replicate them below, ensuring they use self.client and handle errors.
    
    async def simulate_alpha(self, expression: str, region: str = "USA", universe: str = "TOP3000", delay: int = 1, decay: int = 4, neutralization: str = "SUBINDUSTRY", truncation: float = 0.08, test_period: str = "P2Y0M") -> Dict:
        # Construct payload
        sim_payload = {
            "type": "REGULAR",
            "settings": {
                "instrumentType": "EQUITY", "region": region, "universe": universe, "delay": delay,
                "decay": decay, "neutralization": neutralization, "truncation": truncation,
                "testPeriod": test_period, "nanHandling": "OFF", "unitHandling": "VERIFY", "pasteurization": "ON",
                "language": "FASTEXPR", "visualization": False, "maxTrade": "ON"
            },
            "regular": expression
        }
        
        try:
            response = await self._safe_api_call("post", "/simulations", json=sim_payload)
            if response.status_code not in [200, 201, 202]:
                logger.error(f"Brain Simulation Failed [{response.status_code}] | Payload: {json.dumps(sim_payload)} | Response: {response.text}")
                return {"success": False, "error": f"Creation failed: {response.text}"}
            
            location = response.headers.get("Location")
            if not location:
                 location = f"/simulations/{response.json().get('id')}"
                 
            logger.info(f"Simulation started | location={location}")
            return await self._wait_for_simulation(location)
        except Exception as e:
            error = _format_exception(e)
            logger.error(f"Simulate error: {error}")
            return {"success": False, "error": error}

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
    ) -> List[Dict]:
        """
        Simulate multiple alphas in a single batch request (Multi-Simulation).
        Returns a list of results in the same order as expressions.
        """
        if len(expressions) < 2:
            return [
                {
                    "success": False,
                    "error": "Multi-simulation requires at least 2 expressions; single simulation was not submitted",
                }
                for _ in expressions
            ]

        # Construct payload list
        sim_payloads = []
        for expr in expressions:
            sim_payloads.append({
                "type": "REGULAR",
                "settings": {
                    "instrumentType": "EQUITY", "region": region, "universe": universe, "delay": delay,
                    "decay": decay, "neutralization": neutralization, "truncation": truncation,
                    "testPeriod": test_period, "nanHandling": "OFF", "unitHandling": "VERIFY", "pasteurization": "ON",
                    "language": "FASTEXPR", "visualization": False, "maxTrade": "ON"
                },
                "regular": expr
            })
        
        try:
            # POST list of configs
            response = await self._safe_api_call("post", "/simulations", json=sim_payloads)
            
            if response.status_code not in [200, 201, 202]:
                logger.error(f"Batch Simulation Failed [{response.status_code}] | Response: {response.text}")
                # Return failures for all
                return [{"success": False, "error": f"Batch creation failed: {response.text}"} for _ in expressions]
            
            location = response.headers.get("Location")
            if not location:
                # If no location header, check body (unlikely for multi-sim)
                return [{"success": False, "error": "No location header"} for _ in expressions]
                 
            logger.info(f"Batch simulation started | count={len(expressions)} location={location}")
            # Wait for parent simulation
            parent_result = await self._wait_for_multisim(
                location,
                max_wait=max_wait,
                timeout_grace_seconds=timeout_grace_seconds,
                no_child_timeout_seconds=no_child_timeout_seconds,
            )
            
            if not parent_result["success"]:
                return [
                    {
                        "success": False,
                        "error": parent_result.get("error"),
                        "location": parent_result.get("location") or location,
                    }
                    for _ in expressions
                ]
            
            # Map results back to order is tricky if Brain doesn't guarantee order, 
            # but usually 'children' list order might allow correlation if we trust it?
            # Better: match by alpha ID if possible? 
            # Actually ace_lib iterates children and fetches results.
            
            return parent_result["results"]
            
        except Exception as e:
            error = _format_exception(e)
            logger.error(f"Batch Simulate error: {error}")
            return [{"success": False, "error": error} for _ in expressions]

    async def _wait_for_multisim(
        self,
        location: str,
        max_wait: int = 1200,
        timeout_grace_seconds: int = 180,
        no_child_timeout_seconds: int = 0,
    ) -> Dict:
        """
        Poll for multi-simulation completion.
        Reference: ace_lib.py `multisimulation_progress` function.
        Key insight: Use Retry-After header presence to determine if still running.
        """
        # Determine full URL
        if location.startswith("http"):
            poll_url = location
        else:
            poll_url = f"{self.BASE_URL}{location}"
        
        error_flag = False
        retry_count = 0
        max_retries = 3
        start_time = asyncio.get_running_loop().time()
        last_progress_log = start_time
        last_progress = None
        no_child_started_at = None
        last_children_count = 0
        
        while True:
            try:
                elapsed = asyncio.get_running_loop().time() - start_time
                if elapsed > max_wait:
                    timeout_result = await self._collect_multisim_results_if_available(
                        poll_url=poll_url,
                        location=location,
                        grace_seconds=timeout_grace_seconds,
                    )
                    if timeout_result.get("success"):
                        logger.info(
                            "Multi-simulation completed at timeout boundary | "
                            f"location={location}"
                        )
                        return timeout_result
                    return {
                        "success": False,
                        "location": location,
                        "error": (
                            f"Multi-simulation timed out after {int(elapsed)}s "
                            f"(last_progress={last_progress})"
                        ),
                    }

                response = await self._safe_api_call("get", poll_url)
                
                # Handle non-2xx with retry
                if response.status_code // 100 != 2:
                    logger.error(
                        f"Multi-sim poll {poll_url}, Status: {response.status_code}, "
                        f"Response: {response.text[:500]}, Retry"
                    )
                    await asyncio.sleep(30)
                    retry_count += 1
                    if retry_count <= max_retries:
                        continue
                    else:
                        error_flag = True
                        break
                
                # Key check: If Retry-After header is missing or 0, simulation is complete
                retry_after = response.headers.get("Retry-After") or response.headers.get("retry-after")
                progress = None
                children = []
                try:
                    data = response.json()
                    progress = data.get("progress")
                    if progress is not None:
                        last_progress = progress
                    children = data.get("children", []) or []
                    last_children_count = len(children)
                except Exception:
                    pass
                
                if not retry_after or retry_after == "0":
                    # Simulation completed - check for error status
                    data = response.json()
                    if data.get("status", "ERROR") == "ERROR":
                        error_flag = True
                        logger.error(f"Multi-simulation error: {data}")
                    break

                now = asyncio.get_running_loop().time()
                if no_child_timeout_seconds > 0 and not children:
                    if no_child_started_at is None:
                        no_child_started_at = now
                    elif now - no_child_started_at >= no_child_timeout_seconds:
                        return {
                            "success": False,
                            "location": location,
                            "error": (
                                "Multi-simulation no-child timed out after "
                                f"{int(elapsed)}s (last_progress={last_progress}, "
                                f"children={last_children_count})"
                            ),
                        }
                elif children:
                    no_child_started_at = None
                
                # Still running, wait as instructed
                if now - last_progress_log >= 60:
                    logger.info(
                        f"Multi-simulation still running | location={location} "
                        f"elapsed={int(now - start_time)}s retry_after={retry_after} "
                        f"progress={progress if progress is not None else 'unknown'}"
                    )
                    last_progress_log = now
                await asyncio.sleep(float(retry_after))
                
            except Exception as e:
                import traceback
                logger.error(f"Multi-sim poll error: {traceback.format_exc()}")
                await asyncio.sleep(3)
                retry_count += 1
                if retry_count > max_retries:
                    return {"success": False, "error": _format_exception(e)}
        
        # Get children from final response
        try:
            data = response.json()
            children = data.get("children", [])
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to parse multi-sim response from {poll_url}: {_format_exception(e)}",
            }
        
        return await self._fetch_multisim_child_results(
            children=children,
            parent_data=data,
            parent_location=poll_url,
            error_flag=error_flag,
        )

    async def _collect_multisim_results_if_available(
        self,
        poll_url: str,
        location: str,
        grace_seconds: int = 180,
    ) -> Dict:
        """On timeout, do one final parent/child read before declaring failure.

        BRAIN parent multi-simulations often sit at progress=0.35 until the next
        poll, while children may already be COMPLETE and have alpha IDs. The
        normal max_wait boundary should not discard those completed child
        results.
        """
        deadline = asyncio.get_running_loop().time() + max(0, grace_seconds)
        last_status = None
        last_progress = None
        last_children_count = 0
        last_child_error = None
        while True:
            try:
                response = await self._safe_api_call("get", poll_url)
                if response.status_code // 100 != 2:
                    return {
                        "success": False,
                        "error": f"Parent poll failed with status {response.status_code}",
                    }
                data = response.json()
                last_status = data.get("status")
                last_progress = data.get("progress")
                children = data.get("children", []) or []
                last_children_count = len(children)
                if children:
                    result = await self._fetch_multisim_child_results(
                        children=children,
                        parent_data=data,
                        parent_location=location,
                        error_flag=data.get("status") == "ERROR",
                        allow_incomplete=False,
                    )
                    if result.get("success"):
                        return result
                    last_child_error = result.get("error")
                if asyncio.get_running_loop().time() >= deadline:
                    return {
                        "success": False,
                        "error": (
                            "No completed child results before grace deadline "
                            f"(parent_status={last_status}, progress={last_progress}, "
                            f"children={last_children_count}, child_error={last_child_error})"
                        ),
                    }
                await asyncio.sleep(5)
            except Exception as exc:
                logger.warning(
                    "Timeout boundary multi-simulation child fetch failed | "
                    f"location={location} error={_format_exception(exc)}"
                )
                return {"success": False}

    async def _fetch_multisim_child_results(
        self,
        children: List[str],
        parent_data: Dict,
        parent_location: str,
        error_flag: bool = False,
        allow_incomplete: bool = True,
    ) -> Dict:
        """Fetch child simulation results for a completed or nearly-complete parent."""
        if error_flag and not children:
            logger.error(f"Multi-simulation failed: {parent_data}")
            return {
                "success": False,
                "location": parent_location,
                "error": parent_data.get("message") or f"Multi-simulation failed at {parent_location}: {parent_data}",
            }

        if not children:
            logger.warning(f"Multi-simulation completed but no children: {parent_data}")
            return {"success": False, "location": parent_location, "error": "No children in multi-simulation"}

        async def fetch_child_result(child_id: str) -> Dict:
            try:
                child_resp = await self._safe_api_call("get", f"/simulations/{child_id}")
                if child_resp.status_code != 200:
                    logger.error(f"Failed to fetch child sim {child_id}: {child_resp.status_code}")
                    return {"success": False, "error": f"Failed to fetch child {child_id}", "child_id": child_id}

                try:
                    child_data = child_resp.json()
                except Exception:
                    child_data = {"raw_text": child_resp.text[:1000]}

                alpha_id = child_data.get("alpha") if isinstance(child_data, dict) else None
                if alpha_id:
                    result = await self._get_completed_alpha_details(alpha_id)
                    if isinstance(result, dict):
                        result.setdefault("child_id", child_id)
                        result.setdefault("location", parent_location)
                    return result

                retry_after = child_resp.headers.get("Retry-After") or child_resp.headers.get("retry-after")
                status = child_data.get("status") if isinstance(child_data, dict) else None
                if retry_after and status not in {"ERROR", "CANCELLED", "COMPLETE"} and not allow_incomplete:
                    return {
                        "success": False,
                        "_incomplete": True,
                        "error": f"Child {child_id} still running",
                        "child_id": child_id,
                        "raw_response": child_data,
                    }

                if error_flag:
                    logger.error(f"Child simulation {child_id} failed: {child_data}")
                else:
                    logger.warning(f"Child simulation {child_id} has no alpha: {child_data}")
                return {
                    "success": False,
                    "error": self._summarize_simulation_error(child_data),
                    "child_id": child_id,
                    "location": parent_location,
                    "raw_response": child_data,
                }
            except Exception as exc:
                error = _format_exception(exc)
                logger.error(f"Error fetching child {child_id}: {error}")
                return {"success": False, "error": error, "child_id": child_id, "location": parent_location}

        results = list(await asyncio.gather(*(fetch_child_result(str(cid)) for cid in children)))
        if any(result.get("_incomplete") for result in results):
            return {"success": False, "location": parent_location, "error": "Child simulations still running"}
        return {"success": True, "results": results}

    def _summarize_simulation_error(self, data: Any) -> str:
        """Extract the useful BRAIN error text from a child simulation payload."""
        if not isinstance(data, dict):
            return str(data)[:500]

        candidates = [
            data.get("message"),
            data.get("error"),
            data.get("statusMessage"),
            data.get("reason"),
        ]
        for path in (
            ("result", "error"),
            ("result", "message"),
            ("regular", "error"),
            ("regular", "message"),
        ):
            node = data
            for key in path:
                node = node.get(key) if isinstance(node, dict) else None
            candidates.append(node)

        for candidate in candidates:
            if candidate:
                return str(candidate)[:500]

        return json.dumps(data, ensure_ascii=False, default=str)[:500]

    async def _wait_for_simulation(self, location: str, max_wait: int = 900) -> Dict:
        """
        Monitor simulation progress and return result when complete.
        Reference: ace_lib.py `simulation_progress` function.
        Key insight: Use Retry-After header presence to determine if still running.
        """
        # Determine full URL
        if location.startswith("http"):
            poll_url = location
        else:
            poll_url = f"{self.BASE_URL}{location}"
            
        error_flag = False
        retry_count = 0
        max_retries = 3
        start_time = asyncio.get_running_loop().time()
        last_progress_log = start_time
        
        while True:
            try:
                elapsed = asyncio.get_running_loop().time() - start_time
                if elapsed > max_wait:
                    return {
                        "success": False,
                        "error": f"Simulation timed out after {int(elapsed)}s",
                    }

                response = await self._safe_api_call("get", poll_url)
                
                # Handle non-2xx response with retry
                if response.status_code // 100 != 2:
                    logger.error(f"Simulation poll {poll_url}, Status: {response.status_code}, Retry")
                    await asyncio.sleep(30)
                    retry_count += 1
                    if retry_count <= max_retries:
                        continue
                    else:
                        logger.error(f"Simulation {poll_url} failed after {max_retries} retries")
                        error_flag = True
                        break
                
                # Key check: If Retry-After header is missing or 0, simulation is complete
                retry_after = response.headers.get("Retry-After") or response.headers.get("retry-after")
                
                if not retry_after or retry_after == "0":
                    # Simulation completed - check for error status
                    data = response.json()
                    if data.get("status", "ERROR") == "ERROR":
                        error_flag = True
                        logger.error(f"Simulation error: {data}")
                    break
                
                # Still running, wait as instructed
                now = asyncio.get_running_loop().time()
                if now - last_progress_log >= 60:
                    logger.info(
                        f"Simulation still running | location={location} "
                        f"elapsed={int(now - start_time)}s retry_after={retry_after}"
                    )
                    last_progress_log = now
                await asyncio.sleep(float(retry_after))
                
            except Exception as e:
                import traceback
                logger.error(f"Poll loop error: {e}\n{traceback.format_exc()}")
                await asyncio.sleep(3)
                retry_count += 1
                if retry_count > max_retries:
                    return {"success": False, "error": str(e)}
        
        if error_flag:
            try:
                error_data = response.json()
                return {"success": False, "error": error_data.get("message", str(error_data))}
            except:
                return {"success": False, "error": "Simulation failed"}
        
        # Get alpha ID from completed simulation
        try:
            data = response.json()
            alpha_id = data.get("alpha")
            
            if not alpha_id:
                logger.warning(f"Simulation completed but no alpha ID: {data}")
                return {"success": False, "error": "No Alpha ID returned"}
            
            # Fetch full alpha details
            return await self._get_completed_alpha_details(alpha_id)
            
        except Exception as e:
            logger.error(f"Failed to parse simulation result: {e}")
            return {"success": False, "error": str(e)}

    async def _get_completed_alpha_details(self, alpha_id: str, max_wait: float = 180.0) -> Dict:
        """
        Fetch full details for a completed alpha.
        Reference: ace_lib.py `get_simulation_result_json` function.
        Uses retry-after header polling to ensure data is ready.
        
        Real API response structure (from BRAIN MCP):
        - id, type, author, settings, regular{code, description, operatorCount}
        - dateCreated, dateSubmitted, dateModified, name, favorite, hidden
        - stage, status, grade, category, tags, classifications
        - is{pnl, bookSize, longCount, shortCount, turnover, returns, drawdown, margin, sharpe, fitness, startDate, investabilityConstrained{}, riskNeutralized{}, checks[]}
        - os, train, test (same structure as is)
        - prod, competitions, themes, pyramids, pyramidThemes, team, osmosisPoints
        """
        if alpha_id is None:
            return {"success": False, "error": "No alpha ID provided"}
            
        try:
            endpoint = f"/alphas/{alpha_id}"
            
            # Poll until no retry-after header (matching ace_lib.py pattern),
            # but keep it bounded so timeout-boundary child collection cannot
            # hang a mining run indefinitely.
            start_time = asyncio.get_running_loop().time()
            last_progress_log = start_time
            while True:
                elapsed = asyncio.get_running_loop().time() - start_time
                if elapsed > max_wait:
                    return {
                        "success": False,
                        "error": f"Alpha details timed out after {int(elapsed)}s",
                        "alpha_id": alpha_id,
                    }

                response = await self._safe_api_call("get", endpoint)
                
                # Check for retry-after header (case-insensitive)
                retry_after = response.headers.get("Retry-After") or response.headers.get("retry-after")
                
                if retry_after:
                    now = asyncio.get_running_loop().time()
                    if now - last_progress_log >= 60:
                        logger.info(
                            f"Alpha details still pending | alpha_id={alpha_id} "
                            f"elapsed={int(now - start_time)}s retry_after={retry_after}"
                        )
                        last_progress_log = now
                    wait_time = float(retry_after)
                    remaining = max_wait - (asyncio.get_running_loop().time() - start_time)
                    if remaining <= 0:
                        continue
                    await asyncio.sleep(min(wait_time, remaining))
                else:
                    break
            
            if response.status_code != 200:
                logger.error(f"Failed to get alpha details [{response.status_code}]: {response.text}")
                return {"success": False, "error": f"Failed to fetch details: {response.status_code}"}
            
            try:
                alpha = response.json()
            except Exception:
                logger.error(f"Failed to parse alpha JSON: alpha_id={alpha_id}, headers={response.headers}, text={response.text}")
                return {"success": False, "error": "Failed to parse alpha response"}
            
            # Extract stats from each period
            is_stats = alpha.get("is") or {}
            train_stats = alpha.get("train") or {}
            test_stats = alpha.get("test") or {}
            os_stats = alpha.get("os") or {}
            
            # Extract checks from IS stats (important for submission validation)
            checks = is_stats.get("checks", [])
            failed_checks = [c for c in checks if c.get("result") == "FAIL"]
            pending_checks = [c for c in checks if c.get("result") == "PENDING"]
            passed_checks = [c for c in checks if c.get("result") == "PASS"]
            
            # Extract expression from regular.code
            regular = alpha.get("regular") or {}
            expression = regular.get("code")

            return {
                "success": True, 
                "alpha_id": alpha.get("id"),
                "expression": expression,
                "settings": alpha.get("settings", {}),
                "stage": alpha.get("stage"),  # IS or OS
                "status": alpha.get("status"),  # UNSUBMITTED, SUBMITTED, etc.
                "type": alpha.get("type"),  # REGULAR or SUPER
                "dateCreated": alpha.get("dateCreated"),
                "dateSubmitted": alpha.get("dateSubmitted"),
                "classifications": alpha.get("classifications", []),
                
                # Metrics dictionary with all available stats
                "metrics": {
                    # Primary IS metrics (for scoring)
                    "sharpe": is_stats.get("sharpe"),
                    "returns": is_stats.get("returns"),
                    "turnover": is_stats.get("turnover"),
                    "fitness": is_stats.get("fitness"),
                    "drawdown": is_stats.get("drawdown"),  # Renamed from max_dd
                    "pnl": is_stats.get("pnl"),
                    "margin": is_stats.get("margin"),
                    "bookSize": is_stats.get("bookSize"),
                    "longCount": is_stats.get("longCount"),
                    "shortCount": is_stats.get("shortCount"),
                    
                    # Train/Test metrics
                    "train_sharpe": train_stats.get("sharpe"),
                    "train_fitness": train_stats.get("fitness"),
                    "train_turnover": train_stats.get("turnover"),
                    "train_returns": train_stats.get("returns"),
                    "train_drawdown": train_stats.get("drawdown"),
                    
                    "test_sharpe": test_stats.get("sharpe"),
                    "test_fitness": test_stats.get("fitness"),
                    "test_turnover": test_stats.get("turnover"),
                    "test_returns": test_stats.get("returns"),
                    "test_drawdown": test_stats.get("drawdown"),
                    
                    # OS metrics (if available)
                    "os_sharpe": os_stats.get("sharpe") if os_stats else None,
                    "os_fitness": os_stats.get("fitness") if os_stats else None,
                    
                    # Investability and Risk Neutralized stats (nested dicts)
                    "investabilityConstrained": is_stats.get("investabilityConstrained") or {},
                    "riskNeutralized": is_stats.get("riskNeutralized") or {},
                    
                    # Train investability/risk stats
                    "train_investabilityConstrained": train_stats.get("investabilityConstrained") or {},
                    "train_riskNeutralized": train_stats.get("riskNeutralized") or {},
                    
                    # Test investability/risk stats  
                    "test_investabilityConstrained": test_stats.get("investabilityConstrained") or {},
                    "test_riskNeutralized": test_stats.get("riskNeutralized") or {},
                },
                
                # Submission checks (critical for knowing if alpha can be submitted)
                "checks": checks,
                "failed_checks": [c.get("name") for c in failed_checks],
                "pending_checks": [c.get("name") for c in pending_checks],
                "passed_checks": [c.get("name") for c in passed_checks],
                "can_submit": len(failed_checks) == 0 and len(pending_checks) == 0,
                
                # Full period data (for detailed analysis)
                "is": is_stats,
                "os": os_stats,
                "train": train_stats,
                "test": test_stats,
                
                # Additional metadata
                "regular": regular,  # Contains code, description, operatorCount
                "competitions": alpha.get("competitions"),
                "themes": alpha.get("themes"),
                "pyramids": alpha.get("pyramids"),
                
                # Include full raw response for debugging
                "raw": alpha
            }
        except Exception as e:
            logger.error(f"Get alpha details error: {e}")
            return {"success": False, "error": str(e)}

    async def _safe_api_call(self, method: str, endpoint: str, **kwargs) -> httpx.Response:
        """
        Execute API call with auto-reauth on 401 and retry on 429/5xx.
        """
        url = endpoint if endpoint.startswith("http") else f"{self.BASE_URL}{endpoint}"
        retries = 0
        max_retries = 5
        
        while retries < max_retries:
            try:
                response = await getattr(self.client, method.lower())(url, **kwargs)
                
                # 1. Handle 401 Unauthorized (Token Expiry)
                if response.status_code == 401:
                    logger.warning(f"401 Unauthorized for {endpoint}, re-authenticating...")
                    await self._invalidate_session_cache()
                    if await self.authenticate():
                        # Retry immediately with new token
                        response = await getattr(self.client, method.lower())(url, **kwargs)
                
                # 2. Handle 429 Too Many Requests (Rate Limit)
                if response.status_code == 429:
                    retry_after = response.headers.get("Retry-After")
                    wait_time = float(retry_after) if retry_after else (2 ** (retries + 1))
                    logger.warning(f"429 Rate Limit for {endpoint}. Sleeping {wait_time}s (Attempt {retries+1}/{max_retries})")
                    await asyncio.sleep(wait_time)
                    retries += 1
                    continue
                
                # 3. Handle 5xx Server Errors (Temporary Glitch)
                if 500 <= response.status_code < 600:
                    wait_time = 2 ** (retries + 1)
                    logger.warning(f"Server Error {response.status_code} for {endpoint}. Sleeping {wait_time}s")
                    await asyncio.sleep(wait_time)
                    retries += 1
                    continue
                
                return response
                
            except (httpx.RequestError, httpx.TimeoutException) as e:
                # Network level errors
                wait_time = 2 ** (retries + 1)
                logger.error(f"Network error {endpoint}: {e}. Retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
                retries += 1
                
        # If exhausted retries, return the last response or raise
        logger.error(f"Max retries exceeded for {endpoint}")
        if 'response' in locals():
            return response
        raise Exception(f"Failed to connect to {endpoint} after {max_retries} attempts")

    async def get_datasets(self, region: str = "USA", delay: int = 1, universe: str = "TOP3000") -> List[Dict]:
        try:
            response = await self._safe_api_call(
                "GET", "/data-sets",
                params={"region": region, "delay": delay, "universe": universe, "instrumentType": "EQUITY"}
            )
            return response.json().get("results", []) if response.status_code == 200 else []
        except Exception:
            return []

    async def get_datafields(self, dataset_id: str, region: str = "USA", delay: int = 1, universe: str = "TOP3000") -> List[Dict]:
        all_results = []
        offset = 0
        limit = 50
        
        while True:
            try:
                response = await self._safe_api_call(
                    "GET", "/data-fields",
                    params={
                        "dataset.id": dataset_id, 
                        "region": region, 
                        "delay": delay, 
                        "universe": universe, 
                        "instrumentType": "EQUITY",
                        "limit": limit,
                        "offset": offset
                    }
                )
                
                if response.status_code != 200:
                    logger.error(f"Get fields failed: {response.status_code} - {response.text}")
                    break
                    
                data = response.json()
                results = data.get("results", [])
                
                if not results:
                    break
                    
                all_results.extend(results)
                
                if len(results) < limit:
                    break
                    
                offset += limit
                
            except Exception as e:
                logger.error(f"Get fields error: {e}")
                break
                
        return all_results

    async def get_operators(self, detailed: bool = False) -> List[Any]:
        try:
            response = await self._safe_api_call("GET", "/operators")
            if response.status_code == 200:
                data = response.json()
                results = data if isinstance(data, list) else data.get("results", [])
                return results if detailed else [op.get("name") for op in results]
            return self._get_common_operators()
        except Exception:
            return self._get_common_operators()

    async def get_alpha_pnl(self, alpha_id: str) -> Dict:
        try:
            response = await self._safe_api_call("get", f"/alphas/{alpha_id}/recordsets/pnl")
            return response.json() if response.status_code == 200 else {}
        except Exception:
            return {}

    async def check_correlation(self, alpha_id: str, check_type: str = "PROD") -> Dict:
        try:
            endpoint = f"/alphas/{alpha_id}/correlations/{check_type.lower()}"
            response = None
            for _ in range(30):
                response = await self._safe_api_call("get", endpoint)
                if response.status_code != 200:
                    return {}
                retry_after = response.headers.get("Retry-After") or response.headers.get("retry-after")
                if response.text:
                    return response.json()
                if not retry_after or retry_after == "0":
                    return {}
                await asyncio.sleep(float(retry_after))
            return response.json() if response is not None and response.text else {}
        except Exception:
            return {}

    async def get_user_alphas(self, limit: int = 100, offset: int = 0, stage: str = None, search: str = None, start_date: str = None) -> Dict:
        """
        Get user's alphas with pagination.
        endpoint: /users/self/alphas
        """
        try:
            params = {
                "limit": limit, 
                "offset": offset,
                "hidden": False,
                "order": "-dateCreated"
            }
            if stage:
                params["stage"] = stage
            if search:
                params["search"] = search
            if start_date:
                # Brain API often uses 'startDate' for filtering creation date
                params["startDate"] = start_date
                
            response = await self._safe_api_call("GET", "/users/self/alphas", params=params)
            
            if response.status_code == 200:
                return response.json()
            return {"results": [], "count": 0}
        except Exception as e:
            logger.error(f"Failed to get user alphas: {e}")
            return {"results": [], "count": 0}

    def _get_common_operators(self) -> List[str]:
        return ["rank", "ts_rank", "ts_zscore", "ts_mean", "ts_delay", "ts_corr", "ts_max", "ts_min", "abs", "log", "sign"]


# =============================================================================
# Singleton Instance Management
# =============================================================================

_brain_adapter_instance: Optional[BrainAdapter] = None
_brain_adapter_lock = asyncio.Lock()


async def get_brain_adapter() -> BrainAdapter:
    """
    Get or create the singleton BrainAdapter instance.
    
    This provides a standard way to access the adapter throughout the application,
    ensuring session reuse and proper authentication state management.
    
    Returns:
        BrainAdapter instance implementing BrainProtocol
    """
    global _brain_adapter_instance
    
    if _brain_adapter_instance is None:
        async with _brain_adapter_lock:
            if _brain_adapter_instance is None:
                _brain_adapter_instance = BrainAdapter()
                await _brain_adapter_instance.ensure_session()
    
    return _brain_adapter_instance


def get_brain_adapter_sync() -> BrainAdapter:
    """
    Get or create the singleton BrainAdapter instance (sync version).
    
    Warning: This does NOT ensure the session is valid.
    Use get_brain_adapter() in async contexts when possible.
    
    Returns:
        BrainAdapter instance
    """
    global _brain_adapter_instance
    
    if _brain_adapter_instance is None:
        _brain_adapter_instance = BrainAdapter()
    
    return _brain_adapter_instance


# Backward compatibility alias
brain_adapter = get_brain_adapter_sync()


def reset_brain_adapter():
    """Reset the singleton instance. Useful for testing."""
    global _brain_adapter_instance
    _brain_adapter_instance = None
