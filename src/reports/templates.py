"""Canonical report templates: typed metrics -> the exact text format Star was trained on.

The corpus generator renders its cash-flow documents from CASH_FLOW too, so a prompt
built here is indistinguishable from training data.
"""
from dataclasses import dataclass
from string import Formatter

from .credit import DEFAULT_POLICY, CreditAssessment, assess
from .formatting import money, num, pct

CASH_FLOW = {
    "en": {
        "title": "{co} - Cash Flow Analysis, {per}",
        "lines": [
            "Operating cash flow was {ocf}, driven by EBITDA of {ebitda} and a working capital change of {dwc}.",
            "CAPEX totaled {capex} ({capex_pct} of revenue), resulting in free cash flow (FCF) of {fcf}.",
            "FCF conversion, measured as FCF over EBITDA, was {conv}.",
            "The company closed the period with cash of {cash} and net debt of {net_debt}, a leverage ratio of {lev} net debt/EBITDA.",
        ],
        "closings": [
            "Liquidity remains adequate to fund the investment plan without additional borrowing.",
            "The treasury team will prioritize deleveraging before considering share buybacks.",
            "Management expects working capital to normalize over the next two quarters.",
        ],
    },
    "es": {
        "title": "{co} - Análisis de Flujo de Caja, {per}",
        "lines": [
            "El flujo de caja operativo fue de {ocf}, impulsado por un EBITDA de {ebitda} y una variación de capital de trabajo de {dwc}.",
            "El CAPEX sumó {capex} ({capex_pct} de los ingresos), con un flujo de caja libre (FCF) de {fcf}.",
            "La conversión de FCF, medida como FCF sobre EBITDA, fue de {conv}.",
            "La empresa cerró el periodo con caja de {cash} y deuda neta de {net_debt}, un apalancamiento de {lev} deuda neta/EBITDA.",
        ],
        "closings": [
            "La liquidez es suficiente para financiar el plan de inversión sin nueva deuda.",
            "Tesorería priorizará el desapalancamiento antes de evaluar recompras de acciones.",
            "La dirección espera que el capital de trabajo se normalice en los próximos dos trimestres.",
        ],
    },
}

TABLE_LABELS = {
    "en": ["Metric", "Revenue", "EBITDA", "Operating cash flow", "CAPEX", "Working capital change",
           "Free cash flow (FCF)", "Cash", "Net debt", "CAPEX intensity", "FCF conversion", "Net debt/EBITDA", "n/a"],
    "es": ["Métrica", "Ingresos", "EBITDA", "Flujo de caja operativo", "CAPEX",
           "Variación de capital de trabajo", "Flujo de caja libre (FCF)", "Caja", "Deuda neta",
           "Intensidad de CAPEX", "Conversión de FCF", "Deuda neta/EBITDA", "n/d"],
}


@dataclass
class CanonicalReport:
    title: str
    body: str      # computed sentences, figures guaranteed correct
    prompt: str    # title + body in corpus layout; the model continues with the commentary
    fallback: str  # deterministic commentary used when every generated sentence fails verification
    table: str     # Markdown table of inputs and derived ratios


def _context(req):
    m, lang = req.metrics, req.language
    return dict(
        co=req.company_name, per=req.period,
        ocf=money(m.operating_cash_flow, lang), ebitda=money(m.ebitda, lang),
        dwc=money(m.working_capital_change, lang), capex=money(m.capex, lang),
        capex_pct=pct(m.capex_intensity_pct, lang), fcf=money(m.free_cash_flow, lang),
        conv=None if m.fcf_conversion_pct is None else pct(m.fcf_conversion_pct, lang),
        cash=money(m.cash, lang), net_debt=money(m.net_debt, lang),
        lev=None if m.net_leverage is None else num(m.net_leverage, lang) + "x",
    )


def _fields(template):
    return {name for _, name, _, _ in Formatter().parse(template) if name}


def closing_index(net_leverage, free_cash_flow):
    """Which CASH_FLOW closing fits the metrics. Shared by the corpus generator (so the model learns
    the mapping) and the report fallback."""
    if net_leverage is not None and net_leverage > 3:
        return 1  # deleveraging
    return 0 if free_cash_flow >= 0 else 2  # adequate liquidity / working capital normalization


def value_header(lang, currency_label=""):
    if lang == "es":
        return f"Valor (miles de {currency_label})" if currency_label else "Valor (miles)"
    return f"Value (thousands of {currency_label})" if currency_label else "Value (thousands)"


def metrics_table(req, c, currency_label=""):
    m, lang = req.metrics, req.language
    lab = TABLE_LABELS[lang]
    rows = [money(m.revenue, lang), c["ebitda"], c["ocf"], c["capex"], c["dwc"], c["fcf"], c["cash"],
            c["net_debt"], c["capex_pct"], c["conv"] or lab[-1], c["lev"] or lab[-1]]
    lines = [f"| {lab[0]} | {value_header(lang, currency_label)} |", "|---|---:|"]
    lines += [f"| {name} | {value} |" for name, value in zip(lab[1:-1], rows)]
    return "\n".join(lines)


def build_cash_flow(req, currency_label=""):
    t, c = CASH_FLOW[req.language], _context(req)
    title = t["title"].format(**c)
    # Ratios are undefined when EBITDA <= 0: drop the sentences that need them instead of printing "None".
    body = "\n".join(line.format(**c) for line in t["lines"] if all(c[f] is not None for f in _fields(line)))
    return CanonicalReport(
        title=title,
        body=body,
        prompt=f"{title}\n\n{body}\n",
        fallback=t["closings"][closing_index(req.metrics.net_leverage, req.metrics.free_cash_flow)],
        table=metrics_table(req, c, currency_label),
    )


# --- Corporate credit analysis (credit committee opinion) -------------------------------------------------

CREDIT = {
    "en": {
        "title": "{co} - Credit Analysis, {per}",
        "rows": ["Revenue", "EBITDA", "Operating cash flow", "Maintenance CAPEX", "Existing financial debt", "Cash",
                 "Existing annual debt service", "Requested amount", "Annual rate", "Term (months)",
                 "Collateral (adjusted value)"],
        "analysis": {
            "cfads": "Cash flow available for debt service (CFADS) is {cfads}: operating cash flow of {ocf} less "
                     "maintenance CAPEX of {capex}.",
            "service": "The requested facility of {amount} at {rate} over {term} months carries an annual debt "
                       "service of {new_service}, which together with the existing {existing} totals {total_service}.",
            "dscr": "The resulting debt service coverage ratio (DSCR) is {dscr}, leaving a headroom of {headroom}.",
            "leverage": "After disbursement, leverage would be {lev} net debt/EBITDA.",
            "no_leverage": "EBITDA is not positive, so leverage cannot be computed.",
            "collateral": "Collateral offered ({n_collateral}) has an adjusted value of {adjusted}, a coverage of "
                          "{coverage} over the requested amount.",
            "gap": "An additional {gap} of adjusted collateral is needed to reach {min_cov}.",
            "capacity": "Maximum payment capacity at the requested rate and term, keeping a DSCR of {min_dscr}, "
                        "is {max_loan}.",
        },
        "criteria_head": "| Criterion | Value | Threshold | Met |",
        "criteria": {"dscr": "DSCR", "leverage": "Net debt/EBITDA after loan", "coverage": "Collateral coverage"},
        "yes_no": ("Yes", "No"),
        "limits": ("min.", "max."),
        "decision": {
            "favorable": "**Opinion: FAVORABLE.** The facility meets every criterion of the credit policy.",
            "conditional": "**Opinion: FAVORABLE WITH CONDITIONS.** Approval is recommended subject to:",
            "unfavorable": "**Opinion: UNFAVORABLE.** The facility breaches hard limits of the credit policy:",
        },
        "conditions": {
            "reduce_amount": "Reduce the amount to at most {max_loan} to reach a DSCR of {min_dscr}.",
            "restructure_debt": "Restructure existing debt: current debt service leaves no capacity for new debt "
                                "at a DSCR of {min_dscr}.",
            "add_collateral": "Provide additional collateral with an adjusted value of at least {gap}.",
            "leverage_covenant": "Leverage covenant: net debt/EBITDA no higher than {max_lev}, tested quarterly.",
        },
        "reasons": {
            "dscr": "DSCR of {dscr} is below the minimum of {hard_min_dscr}: cash flow does not cover debt service.",
            "leverage": "Leverage of {lev} exceeds the maximum of {hard_max_lev}.",
            "no_ebitda": "EBITDA is not positive.",
        },
        "counteroffer": "Alternative: a facility of up to {max_loan} at the same rate and term would meet the "
                        "minimum DSCR of {min_dscr}.",
        "signoff": "Opinion produced by the credit policy rules; it requires review and signature by the credit "
                   "committee.",
    },
    "es": {
        "title": "{co} - Análisis de Crédito, {per}",
        "rows": ["Ingresos", "EBITDA", "Flujo de caja operativo", "CAPEX de mantenimiento", "Deuda financiera actual",
                 "Caja", "Servicio de deuda actual (anual)", "Monto solicitado", "Tasa anual", "Plazo (meses)",
                 "Garantías (valor ajustado)"],
        "analysis": {
            "cfads": "El flujo disponible para el servicio de deuda (CFADS) es de {cfads}: flujo de caja operativo de "
                     "{ocf} menos CAPEX de mantenimiento de {capex}.",
            "service": "La operación solicitada de {amount} a una tasa de {rate} y {term} meses genera un servicio de "
                       "deuda anual de {new_service}, que sumado al servicio actual de {existing} totaliza "
                       "{total_service}.",
            "dscr": "La cobertura del servicio de deuda (DSCR) resultante es de {dscr}, con un margen de {headroom}.",
            "leverage": "Tras el desembolso, el apalancamiento sería de {lev} deuda neta/EBITDA.",
            "no_leverage": "El EBITDA no es positivo, por lo que el apalancamiento no es calculable.",
            "collateral": "Las garantías ofrecidas ({n_collateral}) suman un valor ajustado de {adjusted}, con una "
                          "cobertura de {coverage} sobre el monto solicitado.",
            "gap": "Faltan {gap} de valor ajustado en garantías para alcanzar {min_cov}.",
            "capacity": "La capacidad de pago máxima a la tasa y plazo solicitados, manteniendo un DSCR de "
                        "{min_dscr}, es de {max_loan}.",
        },
        "criteria_head": "| Criterio | Valor | Umbral | Cumple |",
        "criteria": {"dscr": "DSCR", "leverage": "Deuda neta/EBITDA tras el crédito",
                     "coverage": "Cobertura de garantías"},
        "yes_no": ("Sí", "No"),
        "limits": ("mín.", "máx."),
        "decision": {
            "favorable": "**Dictamen: FAVORABLE.** La operación cumple todos los criterios de la política de crédito.",
            "conditional": "**Dictamen: FAVORABLE CON CONDICIONES.** Se recomienda aprobar sujeto a:",
            "unfavorable": "**Dictamen: DESFAVORABLE.** La operación incumple límites de la política de crédito:",
        },
        "conditions": {
            "reduce_amount": "Reducir el monto a un máximo de {max_loan} para alcanzar un DSCR de {min_dscr}.",
            "restructure_debt": "Reestructurar la deuda existente: el servicio actual no deja capacidad para deuda "
                                "nueva con un DSCR de {min_dscr}.",
            "add_collateral": "Constituir garantías adicionales con un valor ajustado de al menos {gap}.",
            "leverage_covenant": "Covenant de apalancamiento: deuda neta/EBITDA no mayor a {max_lev}, medido "
                                 "trimestralmente.",
        },
        "reasons": {
            "dscr": "El DSCR de {dscr} está por debajo del mínimo de {hard_min_dscr}: el flujo no cubre el servicio "
                    "de deuda.",
            "leverage": "El apalancamiento de {lev} supera el máximo de {hard_max_lev}.",
            "no_ebitda": "El EBITDA no es positivo.",
        },
        "counteroffer": "Alternativa: una operación de hasta {max_loan} a la misma tasa y plazo cumpliría el DSCR "
                        "mínimo de {min_dscr}.",
        "signoff": "Dictamen generado por las reglas de la política de crédito; requiere revisión y firma del "
                   "comité de crédito.",
    },
}


@dataclass
class CreditReport:
    title: str
    table: str
    analysis: str
    criteria: str
    opinion: str
    assessment: CreditAssessment


def _ratio(x, lang, d=2):
    return num(x, lang, d) + "x"


def build_credit(req, currency_label="", policy=DEFAULT_POLICY):
    a, f, fac, lang, t = assess(req, policy), req.financials, req.facility, req.language, CREDIT[req.language]
    m = lambda x: money(x, lang)  # noqa: E731
    c = dict(
        co=req.company_name, per=req.period, cfads=m(a.cfads), ocf=m(f.operating_cash_flow),
        capex=m(f.maintenance_capex), amount=m(fac.amount), rate=num(fac.annual_rate_pct, lang, 2) + "%",
        term=str(fac.term_months), new_service=m(a.new_debt_service), existing=m(f.existing_debt_service),
        total_service=m(a.total_debt_service), dscr=_ratio(a.dscr, lang), headroom=m(a.headroom),
        lev=None if a.leverage_after is None else _ratio(a.leverage_after, lang, 1), adjusted=m(a.adjusted_collateral),
        n_collateral=str(len(req.collateral)), coverage=_ratio(a.coverage, lang), gap=m(a.collateral_gap),
        max_loan=m(a.max_loan), min_dscr=_ratio(policy.min_dscr, lang), hard_min_dscr=_ratio(policy.hard_min_dscr, lang),
        max_lev=_ratio(policy.max_leverage, lang, 1), hard_max_lev=_ratio(policy.hard_max_leverage, lang, 1),
        min_cov=_ratio(policy.min_coverage, lang),
    )
    title = t["title"].format(**c)

    values = [m(f.revenue), m(f.ebitda), c["ocf"], c["capex"], m(f.total_debt), m(f.cash), c["existing"],
              c["amount"], c["rate"], c["term"], c["adjusted"]]
    table = "\n".join([f"| {TABLE_LABELS[lang][0]} | {value_header(lang, currency_label)} |", "|---|---:|"]
                      + [f"| {r} | {v} |" for r, v in zip(t["rows"], values)])

    s = t["analysis"]
    sentences = [s["cfads"], s["service"], s["dscr"], s["leverage"] if c["lev"] else s["no_leverage"],
                 s["collateral"]] + ([s["gap"]] if a.collateral_gap > 0 else []) + [s["capacity"]]
    analysis = " ".join(x.format(**c) for x in sentences)

    # Both limits are shown so every threshold the opinion cites is traceable to this table.
    lo, hi = t["limits"]
    shown = {
        "dscr": (c["dscr"], f"≥ {c['min_dscr']} ({lo} {c['hard_min_dscr']})"),
        "leverage": (c["lev"] or TABLE_LABELS[lang][-1], f"≤ {c['max_lev']} ({hi} {c['hard_max_lev']})"),
        "coverage": (c["coverage"], f"≥ {c['min_cov']}"),
    }
    yes, no = t["yes_no"]
    criteria = "\n".join([t["criteria_head"], "|---|---:|---:|:---:|"] + [
        f"| {t['criteria'][k.key]} | {shown[k.key][0]} | {shown[k.key][1]} | {yes if k.passed else no} |"
        for k in a.criteria])

    lines = [t["decision"][a.decision]]
    if a.decision == "conditional":
        lines += [f"- {t['conditions'][k].format(**c)}" for k in a.conditions]
    elif a.decision == "unfavorable":
        for k in a.criteria:
            if k.hard_fail:
                reason = "no_ebitda" if k.key == "leverage" and a.leverage_after is None else k.key
                lines.append(f"- {t['reasons'][reason].format(**c)}")
        if a.max_loan > 0:
            lines.append("\n" + t["counteroffer"].format(**c))
    opinion = "\n".join(lines) + f"\n\n_{t['signoff']}_"
    return CreditReport(title, table, analysis, criteria, opinion, a)
