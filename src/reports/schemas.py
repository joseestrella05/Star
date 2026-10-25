"""Typed contracts for the structured report endpoints.

Figures are inputs or computed here with Decimal; the model never produces a number
that reaches the report without verification.
"""
from datetime import datetime
from decimal import Decimal
from typing import List, Literal, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

Strict = ConfigDict(extra="forbid")


def _amount(**kwargs):
    return Field(max_digits=14, decimal_places=2, **kwargs)


class CashFlowMetrics(BaseModel):
    """Amounts in USD millions: the only currency and unit the current model was trained on."""
    model_config = Strict

    revenue: Decimal = _amount(gt=0)
    ebitda: Decimal = _amount()
    operating_cash_flow: Decimal = _amount()
    capex: Decimal = _amount(ge=0)
    working_capital_change: Decimal = _amount(default=Decimal(0))
    cash: Decimal = _amount(ge=0)
    net_debt: Decimal = _amount()

    @model_validator(mode="after")
    def check_consistency(self):
        if self.ebitda > self.revenue:
            raise ValueError("ebitda cannot exceed revenue")
        return self

    @computed_field
    @property
    def free_cash_flow(self) -> Decimal:
        return self.operating_cash_flow - self.capex

    @computed_field
    @property
    def capex_intensity_pct(self) -> Decimal:
        return self.capex / self.revenue * 100

    @computed_field
    @property
    def fcf_conversion_pct(self) -> Optional[Decimal]:
        return self.free_cash_flow / self.ebitda * 100 if self.ebitda > 0 else None

    @computed_field
    @property
    def net_leverage(self) -> Optional[Decimal]:
        return self.net_debt / self.ebitda if self.ebitda > 0 else None


class GenerationOptions(BaseModel):
    model_config = Strict

    temperature: float = Field(0.3, ge=0.0, le=1.0)
    max_attempts: int = Field(3, ge=1, le=5, description="retries when figure verification fails")
    seed: Optional[int] = Field(None, description="fixed seed for reproducible, auditable output")


class ReportRequest(BaseModel):
    model_config = Strict

    report_type: Literal["cash_flow"] = "cash_flow"
    # Letters, digits, spaces and basic punctuation only: no markup or special tokens reach the prompt.
    company_name: str = Field(min_length=2, max_length=80, pattern=r"^[\w .,&'()-]+$")
    period: str = Field(pattern=r"^Q[1-4] 20\d{2}$", examples=["Q2 2026"])
    language: Literal["es", "en"] = "es"
    metrics: CashFlowMetrics
    generation: GenerationOptions = Field(default_factory=GenerationOptions)


class ExportRequest(BaseModel):
    model_config = Strict

    markdown: str = Field(min_length=1, max_length=50_000, description="report text, possibly edited by the user")
    format: Literal["md", "pdf"] = "pdf"
    footer: str = Field("", max_length=200, description="e.g. report id, checkpoint hash and date")


class ReportSection(BaseModel):
    key: Literal["header", "table", "analysis", "commentary"]
    markdown: str
    source: Literal["computed", "model", "fallback"]


class Verification(BaseModel):
    passed: bool
    attempts: int
    rejected_figures: List[str]


class ModelInfo(BaseModel):
    checkpoint_sha256: str
    params: int


class ReportResponse(BaseModel):
    report_id: UUID
    created_at: datetime
    request: ReportRequest
    markdown: str
    sections: List[ReportSection]
    verification: Verification
    model: ModelInfo
