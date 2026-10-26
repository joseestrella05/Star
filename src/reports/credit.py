"""Corporate credit analysis: payment capacity, DSCR, leverage, collateral coverage and the committee decision.

Everything here is deterministic Decimal arithmetic plus explicit policy thresholds, so every figure and the
decision itself can be recomputed by an auditor from the request and the policy.
"""
from dataclasses import asdict, dataclass, field
from decimal import Decimal
from typing import Dict, List, Optional

TWELVE = Decimal(12)


@dataclass(frozen=True)
class CreditPolicy:
    """Institution's credit policy. Defaults are common market thresholds; each client sets its own."""
    min_dscr: Decimal = Decimal("1.25")         # favorable at or above
    hard_min_dscr: Decimal = Decimal("1.00")    # unfavorable below: cash flow does not cover debt service
    max_leverage: Decimal = Decimal("3.5")      # net debt / EBITDA after the loan, favorable at or below
    hard_max_leverage: Decimal = Decimal("4.5")  # unfavorable above
    min_coverage: Decimal = Decimal("1.00")     # adjusted collateral / loan amount
    # Share of appraised value NOT counted as collateral (valuation and liquidation risk).
    haircuts: Dict[str, Decimal] = field(default_factory=lambda: {
        "cash_deposit": Decimal("0"),
        "mortgage": Decimal("0.30"),
        "pledge": Decimal("0.50"),
        "personal_guarantee": Decimal("1"),  # recorded, but adds no collateral value
    })

    def as_dict(self):
        return {k: (str(v) if isinstance(v, Decimal) else {t: str(h) for t, h in v.items()})
                for k, v in asdict(self).items()}


DEFAULT_POLICY = CreditPolicy()


def annual_debt_service(amount, annual_rate_pct, term_months):
    """Annual installments of a French (equal-payment) loan: 12 * P * r / (1 - (1 + r)^-n), r monthly."""
    r = Decimal(annual_rate_pct) / 100 / TWELVE
    if r == 0:
        return amount / term_months * TWELVE
    return amount * r / (1 - (1 + r) ** -term_months) * TWELVE


def max_principal(annual_service, annual_rate_pct, term_months):
    """Inverse of annual_debt_service: the largest loan a given annual service can repay."""
    if annual_service <= 0:
        return Decimal(0)
    monthly = annual_service / TWELVE
    r = Decimal(annual_rate_pct) / 100 / TWELVE
    if r == 0:
        return monthly * term_months
    return monthly * (1 - (1 + r) ** -term_months) / r


@dataclass
class Criterion:
    key: str             # "dscr" | "leverage" | "coverage"
    value: Optional[Decimal]
    threshold: Decimal
    passed: bool
    hard_fail: bool      # breaches the hard limit: decision is unfavorable


@dataclass
class CreditAssessment:
    cfads: Decimal                   # cash flow available for debt service = OCF - maintenance CAPEX
    new_debt_service: Decimal
    total_debt_service: Decimal
    dscr: Decimal                    # CFADS / total debt service (service > 0 because amount > 0)
    headroom: Decimal                # CFADS - total debt service
    leverage_after: Optional[Decimal]  # (total debt + loan - cash) / EBITDA; None when EBITDA <= 0
    adjusted_collateral: Decimal
    coverage: Decimal
    collateral_gap: Decimal          # extra adjusted collateral needed to reach min_coverage
    max_loan: Decimal                # largest loan keeping DSCR >= min_dscr at the requested rate and term
    criteria: List[Criterion]
    decision: str                    # "favorable" | "conditional" | "unfavorable"
    conditions: List[str]            # condition keys, rendered by the templates


def assess(req, policy=DEFAULT_POLICY):
    f, fac = req.financials, req.facility
    cfads = f.operating_cash_flow - f.maintenance_capex
    new_service = annual_debt_service(fac.amount, fac.annual_rate_pct, fac.term_months)
    total_service = f.existing_debt_service + new_service
    dscr = cfads / total_service
    leverage = (f.total_debt + fac.amount - f.cash) / f.ebitda if f.ebitda > 0 else None
    adjusted = sum((c.appraised_value * (1 - policy.haircuts[c.type]) for c in req.collateral), Decimal(0))
    coverage = adjusted / fac.amount
    gap = max(fac.amount * policy.min_coverage - adjusted, Decimal(0))
    max_service = cfads / policy.min_dscr - f.existing_debt_service
    max_loan = max_principal(max_service, fac.annual_rate_pct, fac.term_months)

    criteria = [
        Criterion("dscr", dscr, policy.min_dscr, dscr >= policy.min_dscr, dscr < policy.hard_min_dscr),
        Criterion("leverage", leverage, policy.max_leverage,
                  leverage is not None and leverage <= policy.max_leverage,
                  leverage is None or leverage > policy.hard_max_leverage),
        Criterion("coverage", coverage, policy.min_coverage, coverage >= policy.min_coverage, False),
    ]
    if any(c.hard_fail for c in criteria):
        decision, conditions = "unfavorable", []
    elif all(c.passed for c in criteria):
        decision, conditions = "favorable", []
    else:
        decision = "conditional"
        # With no room left for new debt, "reduce the amount to 0" is meaningless: restructure instead.
        reduce = "reduce_amount" if max_loan > 0 else "restructure_debt"
        conditions = [{"dscr": reduce, "leverage": "leverage_covenant", "coverage": "add_collateral"}[c.key]
                      for c in criteria if not c.passed]
    return CreditAssessment(cfads, new_service, total_service, dscr, cfads - total_service, leverage, adjusted,
                            coverage, gap, max_loan, criteria, decision, conditions)
