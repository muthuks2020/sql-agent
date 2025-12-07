"""
Configuration Management for SQL Agent System
Uses Pydantic for validation and type safety
"""
from __future__ import annotations

import os
from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field, SecretStr, field_validator


class DatabaseType(str, Enum):
    """Supported database types"""
    MYSQL = "mysql"
    POSTGRESQL = "postgresql"
    ORACLE = "oracle"
    SQLITE = "sqlite"


class LLMProvider(str, Enum):
    """Supported LLM providers"""
    BEDROCK_CLAUDE = "bedrock_claude"


class LogLevel(str, Enum):
    """Log levels"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class DatabaseConfig(BaseModel):
    """Database connection configuration"""
    db_type: DatabaseType
    host: str = "localhost"
    port: int = 3306
    database: str
    username: Optional[str] = None
    password: Optional[SecretStr] = None
    connection_pool_size: int = Field(default=10, ge=1, le=100)
    connection_timeout: int = Field(default=30, ge=1, le=300)
    query_timeout: int = Field(default=60, ge=1, le=600)
    ssl_enabled: bool = False
    ssl_ca_path: Optional[str] = None
    
    # SQLite specific
    sqlite_path: Optional[str] = None
    
    # Oracle specific
    oracle_service_name: Optional[str] = None
    oracle_sid: Optional[str] = None
    
    @field_validator('port')
    @classmethod
    def validate_port(cls, v: int, info) -> int:
        """Set default port based on database type"""
        return v
    
    def get_default_port(self) -> int:
        """Get default port for database type"""
        ports = {
            DatabaseType.MYSQL: 3306,
            DatabaseType.POSTGRESQL: 5432,
            DatabaseType.ORACLE: 1521,
            DatabaseType.SQLITE: 0,
        }
        return ports.get(self.db_type, 3306)
    
    model_config = {"use_enum_values": True}


class LLMConfig(BaseModel):
    """LLM configuration for Bedrock Claude"""
    provider: LLMProvider = LLMProvider.BEDROCK_CLAUDE
    model_id: str = "anthropic.claude-3-5-sonnet-20241022-v2:0"
    aws_region: str = "us-east-1"
    aws_access_key_id: Optional[SecretStr] = None
    aws_secret_access_key: Optional[SecretStr] = None
    aws_session_token: Optional[SecretStr] = None
    max_tokens: int = Field(default=4096, ge=100, le=100000)
    temperature: float = Field(default=0.1, ge=0.0, le=1.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    retry_attempts: int = Field(default=3, ge=1, le=10)
    retry_delay: float = Field(default=1.0, ge=0.1, le=30.0)
    request_timeout: int = Field(default=120, ge=10, le=600)
    
    model_config = {"use_enum_values": True}


class AgentConfig(BaseModel):
    """Agent-specific configuration"""
    max_iterations: int = Field(default=5, ge=1, le=20)
    enable_self_healing: bool = True
    validation_strictness: str = Field(default="medium", pattern="^(low|medium|high)$")
    include_explanation: bool = True
    parallel_validation: bool = False
    cache_schema: bool = True
    schema_cache_ttl: int = Field(default=300, ge=60, le=3600)


class MetricsConfig(BaseModel):
    """Metrics and monitoring configuration"""
    enabled: bool = True
    export_interval: int = Field(default=60, ge=10, le=600)
    include_latency_histograms: bool = True
    include_token_counts: bool = True
    statsd_host: Optional[str] = None
    statsd_port: int = 8125
    prometheus_port: Optional[int] = None


class LoadTestConfig(BaseModel):
    """Load testing configuration"""
    concurrent_users: int = Field(default=10, ge=1, le=1000)
    spawn_rate: float = Field(default=1.0, ge=0.1, le=100.0)
    run_time: str = "5m"
    target_rps: Optional[float] = None


class SystemConfig(BaseModel):
    """Main system configuration"""
    database: DatabaseConfig
    llm: LLMConfig = Field(default_factory=LLMConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    load_test: LoadTestConfig = Field(default_factory=LoadTestConfig)
    log_level: LogLevel = LogLevel.INFO
    debug_mode: bool = False
    
    @classmethod
    def from_env(cls) -> "SystemConfig":
        """Create configuration from environment variables"""
        db_type = DatabaseType(os.getenv("DB_TYPE", "mysql"))
        
        db_config = DatabaseConfig(
            db_type=db_type,
            host=os.getenv("DB_HOST", "localhost"),
            port=int(os.getenv("DB_PORT", "3306")),
            database=os.getenv("DB_NAME", ""),
            username=os.getenv("DB_USER"),
            password=SecretStr(os.getenv("DB_PASSWORD", "")) if os.getenv("DB_PASSWORD") else None,
            sqlite_path=os.getenv("SQLITE_PATH"),
        )
        
        llm_config = LLMConfig(
            aws_region=os.getenv("AWS_REGION", "us-east-1"),
            model_id=os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-5-sonnet-20241022-v2:0"),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "4096")),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.1")),
        )
        
        return cls(
            database=db_config,
            llm=llm_config,
            log_level=LogLevel(os.getenv("LOG_LEVEL", "INFO")),
            debug_mode=os.getenv("DEBUG_MODE", "false").lower() == "true",
        )
    
    model_config = {"use_enum_values": True}


# Global configuration instance
_config: Optional[SystemConfig] = None


def get_config() -> SystemConfig:
    """Get the global configuration instance"""
    global _config
    if _config is None:
        _config = SystemConfig.from_env()
    return _config


def set_config(config: SystemConfig) -> None:
    """Set the global configuration instance"""
    global _config
    _config = config


def reset_config() -> None:
    """Reset the global configuration instance"""
    global _config
    _config = None
