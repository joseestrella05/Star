#!/usr/bin/env python3
"""Generate a synthetic bilingual (ES/EN) business corpus for pre-training Star.

Every document is built from one internally consistent set of financials, so the
figures in statements, commentary, KPIs and projections agree with each other.
"""
import argparse
import random
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.reports.formatting import money, num, pct  # noqa: E402

EOT = "<|endoftext|>"

COMPANIES = [
    "Andes Capital", "Nova Retail Group", "Helix Pharma", "Atlas Logistics", "Solaris Energy",
    "Quantum Fintech", "Vertex Manufacturing", "Pacífico Foods", "Orion Telecom", "Cobalt Mining",
    "Aurora Software", "Delta Insurance", "Monarca Construcciones", "Brisa Hospitality", "Sigma Health",
    "Ceiba Agro", "Northwind Analytics", "Cóndor Airlines", "Lumen Media", "Tierra Real Estate",
]
PEOPLE = [
    "María González", "James Carter", "Lucía Fernández", "Robert Chen", "Ana Martínez", "David Okafor",
    "Sofía Ramírez", "Emily Watson", "Carlos Herrera", "Priya Nair", "Javier Morales", "Laura Schmidt",
    "Diego Castillo", "Hannah Müller", "Valentina Rojas", "Michael Brooks",
]
ROLES = {
    "en": ["CEO", "CFO", "COO", "CTO", "General Counsel", "Head of Strategy", "Independent Director"],
    "es": ["Director General", "Director Financiero (CFO)", "Director de Operaciones",
           "Director de Tecnología", "Secretario del Consejo", "Director de Estrategia", "Consejero Independiente"],
}
MONTHS = {
    "en": ["January", "February", "March", "April", "May", "June", "July", "August",
           "September", "October", "November", "December"],
    "es": ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto",
           "septiembre", "octubre", "noviembre", "diciembre"],
}
TARGETS = ["Brightline Labs", "Meridian Supply", "Kappa Payments", "Altiplano Data", "Riverstone Clinics"]


def financials(rng):
    rev = rng.uniform(50, 5000)
    growth = rng.gauss(0.08, 0.12)
    cogs = rev * rng.uniform(0.35, 0.65)
    opex = rev * rng.uniform(0.08, 0.25)  # keeps EBITDA margin >= 10%
    da = rev * rng.uniform(0.02, 0.07)
    interest = rev * rng.uniform(0.0, 0.03)
    gross = rev - cogs
    ebitda = gross - opex
    ebit = ebitda - da
    ebt = ebit - interest
    tax = max(ebt, 0) * rng.uniform(0.18, 0.35)
    net = ebt - tax
    capex = rev * rng.uniform(0.03, 0.15)
    dwc = rev * rng.uniform(-0.03, 0.05)
    ocf = ebitda - tax - dwc
    fcf = ocf - capex
    cash = rev * rng.uniform(0.05, 0.4)
    debt = ebitda * rng.uniform(0.5, 4.0)
    return SimpleNamespace(
        rev=rev, prev=rev / (1 + growth), growth=growth * 100, cogs=cogs, opex=opex, da=da,
        interest=interest, gross=gross, ebitda=ebitda, ebit=ebit, ebt=ebt, tax=tax, net=net,
        capex=capex, dwc=dwc, ocf=ocf, fcf=fcf, cash=cash, debt=debt, net_debt=debt - cash,
    )


def period(rng):
    return f"Q{rng.randint(1, 4)} {rng.randint(2019, 2026)}"


def ctx(f, lang, co, per):
    m, p = (lambda x: money(x, lang)), (lambda x: pct(x, lang))
    up = f.growth >= 0
    return dict(
        co=co, per=per, rev=m(f.rev), prev=m(f.prev), growth=p(abs(f.growth)),
        gm=p(f.gross / f.rev * 100), em=p(f.ebitda / f.rev * 100), nm=p(f.net / f.rev * 100),
        ebitda=m(f.ebitda), net=m(f.net), opex=m(f.opex), opex_pct=p(f.opex / f.rev * 100),
        capex=m(f.capex), capex_pct=p(f.capex / f.rev * 100), ocf=m(f.ocf), fcf=m(f.fcf),
        conv=p(f.fcf / f.ebitda * 100), cash=m(f.cash), net_debt=m(f.net_debt),
        lev=num(f.net_debt / f.ebitda, lang) + "x", dwc=m(f.dwc),
        dir={"en": "increased" if up else "decreased", "es": "aumentaron" if up else "disminuyeron"}[lang],
        tone={"en": "solid" if up else "challenging", "es": "sólido" if up else "complejo"}[lang],
    )


COMMENTARY = {
    "en": [
        "Revenue {dir} {growth} YoY to {rev}, compared with {prev} in the prior-year period.",
        "Gross margin stood at {gm}, while EBITDA reached {ebitda}, an EBITDA margin of {em}.",
        "OPEX represented {opex_pct} of revenue as management maintained cost discipline.",
        "Net income was {net}, a net margin of {nm}, reflecting a {tone} operating environment.",
        "Management attributes the result to pricing actions, product mix and operating leverage.",
        "The company reaffirmed its full-year guidance and continues to prioritize margin expansion.",
    ],
    "es": [
        "Los ingresos {dir} {growth} YoY hasta {rev}, frente a {prev} en el mismo periodo del año anterior.",
        "El margen bruto se ubicó en {gm}, mientras que el EBITDA alcanzó {ebitda}, un margen EBITDA de {em}.",
        "El OPEX representó el {opex_pct} de los ingresos gracias a la disciplina en costos.",
        "La utilidad neta fue de {net}, un margen neto de {nm}, en un entorno operativo {tone}.",
        "La dirección atribuye el resultado a ajustes de precios, mezcla de productos y apalancamiento operativo.",
        "La compañía reafirmó su guía anual y mantiene como prioridad la expansión de márgenes.",
    ],
}


def income_statement(rng, lang):
    f, co, per = financials(rng), rng.choice(COMPANIES), period(rng)
    c, m = ctx(f, lang, co, per), (lambda x: money(x, lang))
    if lang == "en":
        head = f"{co} - Consolidated Income Statement, {per} (USD millions)"
        rows = ["Revenue", "Cost of goods sold (COGS)", "Gross profit", "Operating expenses (OPEX)", "EBITDA",
                "Depreciation and amortization", "EBIT", "Interest expense", "Pre-tax income", "Income tax", "Net income"]
        title = "Management commentary:"
    else:
        head = f"{co} - Estado de Resultados Consolidado, {per} (millones de USD)"
        rows = ["Ingresos", "Costo de ventas (COGS)", "Utilidad bruta", "Gastos operativos (OPEX)", "EBITDA",
                "Depreciación y amortización", "EBIT", "Gastos financieros", "Utilidad antes de impuestos",
                "Impuesto sobre la renta", "Utilidad neta"]
        title = "Comentario de la dirección:"
    vals = [f.rev, f.cogs, f.gross, f.opex, f.ebitda, f.da, f.ebit, f.interest, f.ebt, f.tax, f.net]
    table = "\n".join(f"{r}: {m(v)}" for r, v in zip(rows, vals))
    notes = " ".join(t.format(**c) for t in rng.sample(COMMENTARY[lang], rng.randint(3, 5)))
    return f"{head}\n\n{table}\n\n{title} {notes}"


def cash_flow(rng, lang):
    f, co, per = financials(rng), rng.choice(COMPANIES), period(rng)
    c = ctx(f, lang, co, per)
    if lang == "en":
        body = [
            "{co} - Cash Flow Analysis, {per}",
            "",
            "Operating cash flow was {ocf}, driven by EBITDA of {ebitda} and a working capital change of {dwc}.",
            "CAPEX totaled {capex} ({capex_pct} of revenue), resulting in free cash flow (FCF) of {fcf}.",
            "FCF conversion, measured as FCF over EBITDA, was {conv}.",
            "The company closed the period with cash of {cash} and net debt of {net_debt}, a leverage ratio of {lev} net debt/EBITDA.",
            rng.choice([
                "Liquidity remains adequate to fund the investment plan without additional borrowing.",
                "The treasury team will prioritize deleveraging before considering share buybacks.",
                "Management expects working capital to normalize over the next two quarters.",
            ]),
        ]
    else:
        body = [
            "{co} - Análisis de Flujo de Caja, {per}",
            "",
            "El flujo de caja operativo fue de {ocf}, impulsado por un EBITDA de {ebitda} y una variación de capital de trabajo de {dwc}.",
            "El CAPEX sumó {capex} ({capex_pct} de los ingresos), con un flujo de caja libre (FCF) de {fcf}.",
            "La conversión de FCF, medida como FCF sobre EBITDA, fue de {conv}.",
            "La empresa cerró el periodo con caja de {cash} y deuda neta de {net_debt}, un apalancamiento de {lev} deuda neta/EBITDA.",
            rng.choice([
                "La liquidez es suficiente para financiar el plan de inversión sin nueva deuda.",
                "Tesorería priorizará el desapalancamiento antes de evaluar recompras de acciones.",
                "La dirección espera que el capital de trabajo se normalice en los próximos dos trimestres.",
            ]),
        ]
    return "\n".join(body).format(**c)


def board_minutes(rng, lang):
    f, co = financials(rng), rng.choice(COMPANIES)
    m, p = (lambda x: money(x, lang)), (lambda x: pct(x, lang))
    day, month, year = rng.randint(1, 28), rng.choice(MONTHS[lang]), rng.randint(2019, 2026)
    people = rng.sample(PEOPLE, 5)
    roles = rng.sample(ROLES[lang], 5)
    roi, payback = rng.uniform(12, 35), rng.uniform(2, 6)
    dividend = rng.uniform(0.1, 2.5)
    target, price = rng.choice(TARGETS), f.ebitda * rng.uniform(0.3, 1.5)
    votes = rng.choice([(5, 0, 0), (4, 1, 0), (4, 0, 1)])
    attendees = "\n".join(f"- {n}, {r}" for n, r in zip(people, roles))
    if lang == "en":
        items = [
            f"1. Financial results: the CFO presented revenue of {m(f.rev)} and EBITDA of {m(f.ebitda)}, "
            f"with an EBITDA margin of {p(f.ebitda / f.rev * 100)}.",
            f"2. CAPEX budget: the Board reviewed a CAPEX plan of {m(f.capex)} with an expected ROI of {p(roi)} "
            f"and a payback period of {num(payback, lang)} years.",
            f"3. Dividend policy: a dividend of ${num(dividend, lang, 2)} per share was proposed.",
            f"4. M&A: management presented the potential acquisition of {target} for {m(price)}.",
        ]
        res = rng.choice([
            f"RESOLVED: the CAPEX budget of {m(f.capex)} is approved.",
            f"RESOLVED: the dividend of ${num(dividend, lang, 2)} per share is approved, payable next quarter.",
            f"RESOLVED: management is authorized to begin due diligence on {target}.",
        ])
        return (f"Minutes of the Board of Directors Meeting - {co}\nDate: {month} {day}, {year}\n\n"
                f"Attendees:\n{attendees}\n\nAgenda and discussion:\n" + "\n".join(items) +
                f"\n\n{res} Votes in favor: {votes[0]}; against: {votes[1]}; abstentions: {votes[2]}.\n"
                f"There being no further business, the meeting was adjourned.")
    items = [
        f"1. Resultados financieros: el CFO presentó ingresos de {m(f.rev)} y un EBITDA de {m(f.ebitda)}, "
        f"con un margen EBITDA de {p(f.ebitda / f.rev * 100)}.",
        f"2. Presupuesto de CAPEX: el Consejo revisó un plan de CAPEX de {m(f.capex)} con un ROI esperado de {p(roi)} "
        f"y un periodo de recuperación de {num(payback, lang)} años.",
        f"3. Política de dividendos: se propuso un dividendo de ${num(dividend, lang, 2)} por acción.",
        f"4. Fusiones y adquisiciones: la dirección presentó la posible adquisición de {target} por {m(price)}.",
    ]
    res = rng.choice([
        f"ACUERDO: se aprueba el presupuesto de CAPEX por {m(f.capex)}.",
        f"ACUERDO: se aprueba el dividendo de ${num(dividend, lang, 2)} por acción, pagadero el próximo trimestre.",
        f"ACUERDO: se autoriza a la dirección a iniciar la due diligence de {target}.",
    ])
    return (f"Minuta de la Sesión del Consejo de Administración - {co}\nFecha: {day} de {month} de {year}\n\n"
            f"Asistentes:\n{attendees}\n\nOrden del día y discusión:\n" + "\n".join(items) +
            f"\n\n{res} Votos a favor: {votes[0]}; en contra: {votes[1]}; abstenciones: {votes[2]}.\n"
            f"Sin más asuntos que tratar, se levantó la sesión.")


KPIS = [
    # (en, es, low, high, is_pct, higher_is_better)
    ("Revenue growth YoY", "Crecimiento de ingresos YoY", -5, 30, True, True),
    ("Gross margin", "Margen bruto", 25, 70, True, True),
    ("EBITDA margin", "Margen EBITDA", 8, 40, True, True),
    ("ROI", "ROI", 5, 35, True, True),
    ("Customer churn", "Tasa de abandono (churn)", 1, 12, True, False),
    ("Net Promoter Score (NPS)", "Net Promoter Score (NPS)", -10, 75, False, True),
    ("Days sales outstanding (DSO)", "Días de cobro (DSO)", 25, 90, False, False),
    ("CAC payback (months)", "Recuperación del CAC (meses)", 4, 30, False, False),
]
STATUS = {"en": ("On track", "At risk", "Below target"), "es": ("En meta", "En riesgo", "Por debajo de la meta")}


def kpi_report(rng, lang):
    co, per = rng.choice(COMPANIES), period(rng)
    lines, hits = [], 0
    for en, es, lo, hi, is_pct, better_high in rng.sample(KPIS, rng.randint(5, len(KPIS))):
        target, actual = rng.uniform(lo, hi), rng.uniform(lo, hi)
        gap = (actual - target) if better_high else (target - actual)
        status = STATUS[lang][0 if gap >= 0 else 1 if gap > -(hi - lo) * 0.1 else 2]
        hits += gap >= 0
        fmt = (lambda x: pct(x, lang)) if is_pct else (lambda x: num(x, lang))
        vs = "vs. target" if lang == "en" else "vs. meta"
        lines.append(f"- {en if lang == 'en' else es}: {fmt(actual)} {vs} {fmt(target)} ({status})")
    n = len(lines)
    if lang == "en":
        return (f"{co} - KPI Dashboard, {per}\n\n" + "\n".join(lines) +
                f"\n\nSummary: {hits} of {n} KPIs met or exceeded target. "
                + rng.choice(["The executive team will present corrective action plans for KPIs at risk.",
                              "Focus areas for next quarter are retention and collection efficiency.",
                              "Results support the current strategic plan with no changes to guidance."]))
    return (f"{co} - Tablero de KPIs, {per}\n\n" + "\n".join(lines) +
            f"\n\nResumen: {hits} de {n} KPIs cumplieron o superaron la meta. "
            + rng.choice(["El equipo directivo presentará planes de acción para los KPIs en riesgo.",
                          "Las prioridades del próximo trimestre son la retención y la eficiencia en cobranza.",
                          "Los resultados respaldan el plan estratégico vigente sin cambios en la guía."]))


def dcf(base_rev, cagr, margin0, margin_step, capex_pct, tax, wacc, g, years):
    """Return per-year rows and enterprise value (Gordon growth terminal value)."""
    rows, pv, rev = [], 0.0, base_rev
    for t in range(1, years + 1):
        rev *= 1 + cagr
        ebitda = rev * (margin0 + margin_step * t)
        fcf = ebitda * (1 - tax) - rev * capex_pct
        pv += fcf / (1 + wacc) ** t
        rows.append((rev, ebitda, fcf))
    tv = rows[-1][2] * (1 + g) / (wacc - g)
    return rows, pv + tv / (1 + wacc) ** years


def projection(rng, lang):
    f, co = financials(rng), rng.choice(COMPANIES)
    m, p = (lambda x: money(x, lang)), (lambda x: pct(x, lang))
    base_year, years = rng.randint(2022, 2026), rng.randint(3, 5)
    cagr, margin0, step = rng.uniform(0.03, 0.2), f.ebitda / f.rev, rng.uniform(0.0, 0.015)
    capex_pct, tax = f.capex / f.rev, rng.uniform(0.2, 0.3)
    wacc, g = rng.uniform(0.08, 0.13), rng.uniform(0.015, 0.03)
    rows, ev = dcf(f.rev, cagr, margin0, step, capex_pct, tax, wacc, g, years)
    if lang == "en":
        table = "\n".join(f"{base_year + i + 1}E: Revenue {m(r)} | EBITDA {m(e)} | FCF {m(c)}"
                          for i, (r, e, c) in enumerate(rows))
        return (f"{co} - {years}-Year Financial Projection (base year {base_year})\n\n"
                f"Assumptions: revenue CAGR of {p(cagr * 100)}, starting EBITDA margin of {p(margin0 * 100)} "
                f"expanding {p(step * 100)} per year, CAPEX at {p(capex_pct * 100)} of revenue, "
                f"tax rate of {p(tax * 100)}, WACC of {p(wacc * 100)} and terminal growth of {p(g * 100)}.\n\n"
                f"{table}\n\nDiscounting projected FCF and the terminal value at the WACC yields an "
                f"enterprise value of {m(ev)}, or {num(ev / f.ebitda, lang)}x current EBITDA.")
    table = "\n".join(f"{base_year + i + 1}E: Ingresos {m(r)} | EBITDA {m(e)} | FCF {m(c)}"
                      for i, (r, e, c) in enumerate(rows))
    return (f"{co} - Proyección Financiera a {years} Años (año base {base_year})\n\n"
            f"Supuestos: CAGR de ingresos de {p(cagr * 100)}, margen EBITDA inicial de {p(margin0 * 100)} "
            f"con expansión de {p(step * 100)} anual, CAPEX de {p(capex_pct * 100)} de los ingresos, "
            f"tasa impositiva de {p(tax * 100)}, WACC de {p(wacc * 100)} y crecimiento terminal de {p(g * 100)}.\n\n"
            f"{table}\n\nAl descontar el FCF proyectado y el valor terminal al WACC se obtiene un "
            f"valor empresa de {m(ev)}, equivalente a {num(ev / f.ebitda, lang)}x el EBITDA actual.")


GENERATORS = [income_statement, cash_flow, board_minutes, kpi_report, projection]


def generate_corpus(n_docs, seed=1337):
    rng = random.Random(seed)
    return [rng.choice(GENERATORS)(rng, rng.choice(("en", "es"))) for _ in range(n_docs)]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--docs", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out", type=Path, default=Path("data/business_corpus.txt"))
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    text = f"\n{EOT}\n".join(generate_corpus(args.docs, args.seed))
    args.out.write_text(text, encoding="utf-8")
    print(f"{args.docs} docs, {len(text) / 1e6:.2f}M chars -> {args.out}")


if __name__ == "__main__":
    main()
