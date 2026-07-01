"""Source config schema (literature source registry)."""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from .common import new_id, utcnow


class SourceConfigDoc(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str = Field(default_factory=new_id)
    source_name: str
    enabled: bool = False
    requires_api_key: bool = False
    status_message: Optional[str] = None
    last_success_at: Optional[datetime] = None
    last_error_at: Optional[datetime] = None
    last_error_message: Optional[str] = None
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)


class SourceConfigPatch(BaseModel):
    enabled: Optional[bool] = None
    status_message: Optional[str] = None
