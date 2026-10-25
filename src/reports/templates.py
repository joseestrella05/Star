"""Canonical report templates: typed metrics -> the exact text format Star was trained on.

The corpus generator renders its cash-flow documents from CASH_FLOW too, so a prompt
built here is indistinguishable from training data.
"""
from dataclasses import dataclass
from string import Formatter

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
    "en": ["Metric", "Value", "Revenue", "EBITDA", "Operating cash flow", "CAPEX", "Working capital change",
           "Free cash flow (FCF)", "Cash", "Net debt", "CAPEX intensity", "FCF conversion", "Net debt/EBITDA", "n/a"],
    "es": ["Métrica", "Valor", "Ingresos", "EBITDA", "Flujo de caja operativo", "CAPEX",
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


def _fallback_index(m):
    if m.net_leverage is not None and m.net_leverage > 3:
        return 1  # deleveraging
    return 0 if m.free_cash_flow >= 0 else 2  # adequate liquidity / working capital normalization


def metrics_table(req, c):
    m, lang = req.metrics, req.language
    lab = TABLE_LABELS[lang]
    rows = [money(m.revenue, lang), c["ebitda"], c["ocf"], c["capex"], c["dwc"], c["fcf"], c["cash"],
            c["net_debt"], c["capex_pct"], c["conv"] or lab[-1], c["lev"] or lab[-1]]
    lines = [f"| {lab[0]} | {lab[1]} |", "|---|---:|"]
    lines += [f"| {name} | {value} |" for name, value in zip(lab[2:-1], rows)]
    return "\n".join(lines)


def build_cash_flow(req):
    t, c = CASH_FLOW[req.language], _context(req)
    title = t["title"].format(**c)
    # Ratios are undefined when EBITDA <= 0: drop the sentences that need them instead of printing "None".
    body = "\n".join(line.format(**c) for line in t["lines"] if all(c[f] is not None for f in _fields(line)))
    return CanonicalReport(
        title=title,
        body=body,
        prompt=f"{title}\n\n{body}\n",
        fallback=t["closings"][_fallback_index(req.metrics)],
        table=metrics_table(req, c),
    )
