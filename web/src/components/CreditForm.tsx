import type { CollateralType, Lang } from "../api";
import { Button, Card, Field, Input, Select } from "./ui";

const AMOUNT = /^-?\d{1,12}([.,]\d{1,2})?$/;
const NON_NEG = /^\d{1,12}([.,]\d{1,2})?$/;

export type CreditFormState = {
  company_name: string;
  period: string;
  language: Lang;
  financials: {
    revenue: string; ebitda: string; operating_cash_flow: string; maintenance_capex: string;
    total_debt: string; cash: string; existing_debt_service: string;
  };
  facility: { amount: string; annual_rate_pct: string; term_months: string };
  collateral: { type: CollateralType; description: string; appraised_value: string }[];
};

const COLLATERAL_LABELS: Record<CollateralType, string> = {
  mortgage: "Hipoteca",
  pledge: "Prenda",
  cash_deposit: "Depósito en garantía",
  personal_guarantee: "Fianza personal",
};

const YEARS = [2023, 2024, 2025, 2026, 2027];
const PERIODS = YEARS.flatMap((y) => [1, 2, 3, 4].map((q) => `Q${q} ${y}`)).reverse();

export const EMPTY_CREDIT: CreditFormState = {
  company_name: "",
  period: "Q2 2026",
  language: "es",
  financials: { revenue: "", ebitda: "", operating_cash_flow: "", maintenance_capex: "0", total_debt: "", cash: "", existing_debt_service: "" },
  facility: { amount: "", annual_rate_pct: "10", term_months: "60" },
  collateral: [],
};

export function validateCredit(f: CreditFormState): Record<string, string> {
  const errs: Record<string, string> = {};
  if (f.company_name.trim().length < 2) errs.company_name = "Mínimo 2 caracteres";

  const fKeys: (keyof typeof f.financials)[] = ["revenue", "ebitda", "operating_cash_flow", "maintenance_capex", "total_debt", "cash", "existing_debt_service"];
  for (const k of fKeys) {
    const v = f.financials[k].trim();
    if (!AMOUNT.test(v)) { errs[`fin_${k}`] = v ? "Número inválido" : "Requerido"; continue; }
    const n = Number(v.replace(",", "."));
    if (k === "revenue" && n <= 0) errs[`fin_${k}`] = "Debe ser mayor que 0";
    if (["maintenance_capex", "total_debt", "cash", "existing_debt_service"].includes(k) && n < 0) errs[`fin_${k}`] = "No puede ser negativo";
  }
  if (!errs.fin_revenue && !errs.fin_ebitda) {
    const rev = Number(f.financials.revenue.replace(",", "."));
    const ebi = Number(f.financials.ebitda.replace(",", "."));
    if (ebi > rev) errs.fin_ebitda = "No puede superar los ingresos";
  }

  const { amount, annual_rate_pct, term_months } = f.facility;
  if (!NON_NEG.test(amount.trim()) || Number(amount.replace(",", ".")) <= 0) errs.fac_amount = "Debe ser mayor que 0";
  const rate = Number(annual_rate_pct.replace(",", "."));
  if (!NON_NEG.test(annual_rate_pct.trim()) || rate < 0 || rate > 100) errs.fac_rate = "Tasa entre 0 y 100";
  const months = parseInt(term_months);
  if (!term_months.trim() || isNaN(months) || months < 1 || months > 360) errs.fac_months = "Plazo 1–360 meses";

  f.collateral.forEach((c, i) => {
    if (c.description.trim().length < 2) errs[`col_desc_${i}`] = "Mínimo 2 caracteres";
    if (!NON_NEG.test(c.appraised_value.trim())) errs[`col_val_${i}`] = "Valor inválido";
  });
  return errs;
}

/** Build the API request body from form state. */
export function buildCreditRequest(f: CreditFormState) {
  const n = (s: string) => s.trim().replace(",", ".");
  return {
    report_type: "credit_analysis" as const,
    company_name: f.company_name.trim(),
    period: f.period,
    language: f.language,
    financials: Object.fromEntries(
      Object.entries(f.financials).map(([k, v]) => [k, n(v)])
    ),
    facility: { amount: n(f.facility.amount), annual_rate_pct: n(f.facility.annual_rate_pct), term_months: parseInt(f.facility.term_months) },
    collateral: f.collateral.map((c) => ({ type: c.type, description: c.description.trim(), appraised_value: n(c.appraised_value) })),
  };
}

export function CreditForm({
  form,
  onChange,
  errors,
  currencyLabel,
}: {
  form: CreditFormState;
  onChange: (f: CreditFormState) => void;
  errors: Record<string, string>;
  currencyLabel: string;
}) {
  const setFin = (k: keyof typeof form.financials, v: string) =>
    onChange({ ...form, financials: { ...form.financials, [k]: v } });
  const setFac = (k: keyof typeof form.facility, v: string) =>
    onChange({ ...form, facility: { ...form.facility, [k]: v } });

  function addCollateral() {
    onChange({ ...form, collateral: [...form.collateral, { type: "mortgage", description: "", appraised_value: "" }] });
  }
  function removeCollateral(i: number) {
    onChange({ ...form, collateral: form.collateral.filter((_, j) => j !== i) });
  }
  function setCol(i: number, patch: Partial<typeof form.collateral[0]>) {
    onChange({ ...form, collateral: form.collateral.map((c, j) => (j === i ? { ...c, ...patch } : c)) });
  }

  const unit = currencyLabel ? `Miles de ${currencyLabel}` : "En miles";

  return (
    <div className="space-y-5">
      <Card title="Solicitante">
        <div className="grid gap-4 sm:grid-cols-[1fr_auto_auto]">
          <Field label="Empresa" error={errors.company_name}>
            <Input value={form.company_name} invalid={!!errors.company_name} maxLength={80}
              onChange={(e) => onChange({ ...form, company_name: e.target.value })} placeholder="Constructora Caribe" />
          </Field>
          <Field label="Trimestre">
            <Select value={form.period} onChange={(e) => onChange({ ...form, period: e.target.value })}>
              {PERIODS.map((p) => <option key={p}>{p}</option>)}
            </Select>
          </Field>
          <Field label="Idioma">
            <Select value={form.language} onChange={(e) => onChange({ ...form, language: e.target.value as Lang })}>
              <option value="es">Español</option>
              <option value="en">English</option>
            </Select>
          </Field>
        </div>
      </Card>

      <Card title="Financieros del solicitante" aside={<span className="text-xs text-muted">{unit}</span>}>
        <div className="grid gap-4 sm:grid-cols-2">
          {(["revenue", "ebitda", "operating_cash_flow", "maintenance_capex", "total_debt", "cash", "existing_debt_service"] as const).map((k) => (
            <Field key={k} label={FIN_LABELS[k]} error={errors[`fin_${k}`]}>
              <Input inputMode="decimal" value={form.financials[k]} invalid={!!errors[`fin_${k}`]}
                className="text-right tabular-nums" onChange={(e) => setFin(k, e.target.value)} placeholder="0.00" />
            </Field>
          ))}
        </div>
      </Card>

      <Card title="Facilidad crediticia solicitada" aside={<span className="text-xs text-muted">{unit}</span>}>
        <div className="grid gap-4 sm:grid-cols-3">
          <Field label="Monto solicitado" error={errors.fac_amount}>
            <Input inputMode="decimal" value={form.facility.amount} invalid={!!errors.fac_amount}
              className="text-right tabular-nums" onChange={(e) => setFac("amount", e.target.value)} placeholder="0.00" />
          </Field>
          <Field label="Tasa anual (%)" error={errors.fac_rate}>
            <Input inputMode="decimal" value={form.facility.annual_rate_pct} invalid={!!errors.fac_rate}
              className="text-right tabular-nums" onChange={(e) => setFac("annual_rate_pct", e.target.value)} placeholder="10.00" />
          </Field>
          <Field label="Plazo (meses)" error={errors.fac_months}>
            <Input inputMode="numeric" value={form.facility.term_months} invalid={!!errors.fac_months}
              className="text-right tabular-nums" onChange={(e) => setFac("term_months", e.target.value)} placeholder="60" />
          </Field>
        </div>
      </Card>

      <Card title="Garantías" aside={
        <Button variant="secondary" type="button" onClick={addCollateral} className="h-8 px-3 text-xs">+ Agregar garantía</Button>
      }>
        {form.collateral.length === 0 ? (
          <p className="text-sm text-muted">Sin garantías. El análisis se realizará sin cobertura colateral.</p>
        ) : (
          <div className="space-y-4">
            {form.collateral.map((c, i) => (
              <div key={i} className="grid gap-3 rounded-lg border border-line p-3 sm:grid-cols-[auto_1fr_1fr_auto]">
                <Field label="Tipo">
                  <Select value={c.type} onChange={(e) => setCol(i, { type: e.target.value as CollateralType })}>
                    {(Object.keys(COLLATERAL_LABELS) as CollateralType[]).map((t) => (
                      <option key={t} value={t}>{COLLATERAL_LABELS[t]}</option>
                    ))}
                  </Select>
                </Field>
                <Field label="Descripción" error={errors[`col_desc_${i}`]}>
                  <Input value={c.description} invalid={!!errors[`col_desc_${i}`]} maxLength={80}
                    onChange={(e) => setCol(i, { description: e.target.value })} placeholder="Nave industrial Zona Norte" />
                </Field>
                <Field label="Valor tasado" error={errors[`col_val_${i}`]}>
                  <Input inputMode="decimal" value={c.appraised_value} invalid={!!errors[`col_val_${i}`]}
                    className="text-right tabular-nums" onChange={(e) => setCol(i, { appraised_value: e.target.value })} placeholder="0.00" />
                </Field>
                <div className="flex items-end pb-0.5">
                  <button type="button" aria-label="Eliminar garantía"
                    className="flex size-10 items-center justify-center rounded-lg border border-line text-muted hover:border-red-300 hover:text-red-600"
                    onClick={() => removeCollateral(i)}>✕</button>
                </div>
              </div>
            ))}
          </div>
        )}
      </Card>
    </div>
  );
}

const FIN_LABELS: Record<keyof CreditFormState["financials"], string> = {
  revenue: "Ingresos",
  ebitda: "EBITDA",
  operating_cash_flow: "Flujo de caja operativo (post-impuesto)",
  maintenance_capex: "CAPEX de mantenimiento",
  total_debt: "Deuda financiera existente",
  cash: "Caja",
  existing_debt_service: "Servicio de deuda existente (anual)",
};
