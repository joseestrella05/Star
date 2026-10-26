"""Typed contracts for the structured report endpoints.

Figures are inputs or computed here with Decimal; the model never produces a number
that reaches the report without verification.
"""
from datetime import datetime
from decimal import Decimal
from typing import Annotated, Any, Dict, List, Literal, Optional, Union
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Discriminator, Field, Tag, computed_field, model_validator

Strict = ConfigDict(extra="forbid")


def _amount(**kwargs):
    return Field(max_digits=14, decimal_places=2, **kwargs)


class CashFlowMetrics(BaseModel):
    """Amounts in thousands of the institution's currency (set by the core banking system, not here)."""
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


# Letters, digits, spaces and basic punctuation only: no markup or special tokens reach a prompt or a PDF.
SAFE_TEXT = r"^[\w .,&'()-]+$"


class _ReportBase(BaseModel):
    model_config = Strict

    company_name: str = Field(min_length=2, max_length=80, pattern=SAFE_TEXT)
    period: str = Field(pattern=r"^Q[1-4] 20\d{2}$", examples=["Q2 2026"])
    language: Literal["es", "en"] = "es"


class ReportRequest(_ReportBase):
    """Cash-flow analysis: computed figures plus model commentary."""
    report_type: Literal["cash_flow"] = "cash_flow"
    metrics: CashFlowMetrics
    generation: GenerationOptions = Field(default_factory=GenerationOptions)


class CreditFinancials(BaseModel):
    """Borrower's last-twelve-months figures, in thousands."""
    model_config = Strict

    revenue: Decimal = _amount(gt=0)
    ebitda: Decimal = _amount()
    operating_cash_flow: Decimal = _amount(description="after taxes")
    maintenance_capex: Decimal = _amount(ge=0)
    total_debt: Decimal = _amount(ge=0, description="existing financial debt")
    cash: Decimal = _amount(ge=0)
    existing_debt_service: Decimal = _amount(ge=0, description="annual principal + interest on existing debt")

    @model_validator(mode="after")
    def check_consistency(self):
        if self.ebitda > self.revenue:
            raise ValueError("ebitda cannot exceed revenue")
        return self


class CreditFacility(BaseModel):
    """Requested loan, repaid in equal monthly installments (French amortization)."""
    model_config = Strict

    amount: Decimal = _amount(gt=0)
    annual_rate_pct: Decimal = Field(ge=0, le=100, max_digits=5, decimal_places=2)
    term_months: int = Field(ge=1, le=360)


CollateralType = Literal["mortgage", "pledge", "cash_deposit", "personal_guarantee"]


class Collateral(BaseModel):
    model_config = Strict

    type: CollateralType
    description: str = Field(min_length=2, max_length=80, pattern=SAFE_TEXT)
    appraised_value: Decimal = _amount(ge=0)


class CreditAnalysisRequest(_ReportBase):
    """Corporate credit analysis for the credit committee. Fully deterministic: no model text."""
    report_type: Literal["credit_analysis"]
    financials: CreditFinancials
    facility: CreditFacility
    collateral: List[Collateral] = Field(default_factory=list, max_length=20)


def _report_kind(value):
    kind = value.get("report_type") if isinstance(value, dict) else getattr(value, "report_type", None)
    return kind or "cash_flow"  # requests predating report types are cash-flow reports


AnyReportRequest = Annotated[
    Union[Annotated[ReportRequest, Tag("cash_flow")], Annotated[CreditAnalysisRequest, Tag("credit_analysis")]],
    Discriminator(_report_kind),
]


class ExportRequest(BaseModel):
    model_config = Strict

    markdown: str = Field(min_length=1, max_length=50_000, description="report text, possibly edited by the user")
    format: Literal["md", "pdf"] = "pdf"
    footer: str = Field("", max_length=200, description="e.g. report id, checkpoint hash and date")


class ReportSection(BaseModel):
    key: Literal["header", "table", "analysis", "commentary", "criteria", "opinion"]
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
    request: AnyReportRequest
    markdown: str
    sections: List[ReportSection]
    verification: Verification
    model: ModelInfo
    policy: Optional[Dict[str, Any]] = Field(None, description="credit policy thresholds applied (credit analysis)")
