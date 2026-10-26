import { useRef, useState, type DragEvent } from "react";
import { api, ApiError, METRIC_KEYS, type MetricKey, type ReportRequest } from "../api";
import { Button, Card, Field, Input, Select } from "./ui";

export type FormState = Pick<ReportRequest, "company_name" | "period" | "language" | "metrics">;

const METRIC_LABELS: Record<MetricKey, string> = {
  revenue: "Ingresos",
  ebitda: "EBITDA",
  operating_cash_flow: "Flujo de caja operativo",
  capex: "CAPEX",
  working_capital_change: "Variación de capital de trabajo",
  cash: "Caja",
  net_debt: "Deuda neta",
};

// Decimal comma or point, no thousands separators ("1.234,5" is ambiguous in a single field).
const AMOUNT = /^-?\d{1,12}([.,]\d{1,2})?$/;

/** Metrics as the API expects them: plain decimal strings with a point. */
export function normalizeMetrics(m: FormState["metrics"]): FormState["metrics"] {
  return Object.fromEntries(METRIC_KEYS.map((k) => [k, m[k].trim().replace(",", ".")])) as FormState["metrics"];
}
const YEARS = [2023, 2024, 2025, 2026, 2027];
const PERIODS = YEARS.flatMap((y) => [1, 2, 3, 4].map((q) => `Q${q} ${y}`)).reverse();

/** Client-side checks mirror the API schema so most errors show before submitting. */
export function validate(form: FormState): Partial<Record<string, string>> {
  const errors: Record<string, string> = {};
  if (form.company_name.trim().length < 2) errors.company_name = "Mínimo 2 caracteres";
  for (const key of METRIC_KEYS) {
    const v = form.metrics[key].trim();
    if (!AMOUNT.test(v)) errors[key] = v ? "Número sin separador de miles, máximo 2 decimales" : "Requerido";
  }
  const m = normalizeMetrics(form.metrics);
  if (Number(m.revenue) <= 0) errors.revenue ??= "Debe ser mayor que 0";
  for (const k of ["capex", "cash"] as const) if (Number(m[k]) < 0) errors[k] ??= "No puede ser negativo";
  if (!errors.ebitda && Number(m.ebitda) > Number(m.revenue)) errors.ebitda = "No puede superar los ingresos";
  return errors;
}

function ratioPreview(raw: FormState["metrics"]) {
  if (METRIC_KEYS.some((k) => !AMOUNT.test(raw[k].trim()))) return null;
  const m = normalizeMetrics(raw);
  const n = (k: MetricKey) => Number(m[k]);
  const fcf = n("operating_cash_flow") - n("capex");
  const ebitda = n("ebitda");
  const fmt = (x: number, d = 1) => x.toLocaleString("es-ES", { minimumFractionDigits: d, maximumFractionDigits: d });
  return [
    ["FCF", fmt(fcf, 0)],
    ["Conversión de FCF", ebitda > 0 ? `${fmt((fcf / ebitda) * 100)}%` : "n/d"],
    ["Deuda neta/EBITDA", ebitda > 0 ? `${fmt(n("net_debt") / ebitda)}x` : "n/d"],
    ["Intensidad de CAPEX", n("revenue") > 0 ? `${fmt((n("capex") / n("revenue")) * 100)}%` : "n/d"],
  ];
}

export function MetricsForm({
  form,
  onChange,
  errors,
  currencyLabel,
}: {
  form: FormState;
  onChange: (f: FormState) => void;
  errors: Partial<Record<string, string>>;
  currencyLabel: string;
}) {
  const fileInput = useRef<HTMLInputElement>(null);
  const [upload, setUpload] = useState<{ busy: boolean; messages: string[]; ok?: string }>({ busy: false, messages: [] });
  const setMetric = (key: MetricKey, value: string) => onChange({ ...form, metrics: { ...form.metrics, [key]: value } });

  async function handleFile(file: File | undefined) {
    if (!file) return;
    setUpload({ busy: true, messages: [] });
    try {
      const { fields, metrics } = await api.ingest(file);
      const picked = Object.fromEntries(METRIC_KEYS.map((k) => [k, String(metrics[k])])) as FormState["metrics"];
      onChange({ ...form, ...fields, metrics: picked });
      setUpload({ busy: false, messages: [], ok: `Datos cargados desde ${file.name}` });
    } catch (e) {
      setUpload({ busy: false, messages: e instanceof ApiError ? e.messages : ["No se pudo leer el archivo"] });
    }
  }

  const onDrop = (e: DragEvent) => {
    e.preventDefault();
    handleFile(e.dataTransfer.files[0]);
  };
  const ratios = ratioPreview(form.metrics);

  return (
    <div className="space-y-5">
      <Card title="Periodo">
        <div className="grid gap-4 sm:grid-cols-[1fr_auto_auto]">
          <Field label="Empresa" error={errors.company_name}>
            <Input value={form.company_name} invalid={!!errors.company_name} maxLength={80}
              onChange={(e) => onChange({ ...form, company_name: e.target.value })} placeholder="Cooperativa Andina" />
          </Field>
          <Field label="Trimestre">
            <Select value={form.period} onChange={(e) => onChange({ ...form, period: e.target.value })}>
              {PERIODS.map((p) => <option key={p}>{p}</option>)}
            </Select>
          </Field>
          <Field label="Idioma del informe">
            <Select value={form.language} onChange={(e) => onChange({ ...form, language: e.target.value as FormState["language"] })}>
              <option value="es">Español</option>
              <option value="en">English</option>
            </Select>
          </Field>
        </div>
      </Card>

      <Card title="Métricas" aside={<span className="text-xs text-muted">{currencyLabel ? `Miles de ${currencyLabel}` : "En miles"}</span>}>
        <div
          onDragOver={(e) => e.preventDefault()}
          onDrop={onDrop}
          className="mb-5 flex flex-col items-start gap-3 rounded-lg border border-dashed border-slate-300 bg-slate-50 p-4 sm:flex-row sm:items-center sm:justify-between"
        >
          <p className="text-sm text-muted">
            Arrastre un CSV o Excel con la plantilla, o ingrese los datos a mano.{" "}
            <a className="font-medium text-brand underline underline-offset-2" href={api.templateUrl}>Descargar plantilla</a>
          </p>
          <Button variant="secondary" type="button" disabled={upload.busy} onClick={() => fileInput.current?.click()}>
            {upload.busy ? "Leyendo…" : "Subir archivo"}
          </Button>
          <input ref={fileInput} type="file" hidden accept=".csv,.xlsx,text/csv"
            onChange={(e) => { handleFile(e.target.files?.[0]); e.target.value = ""; }} />
        </div>
        {upload.ok && <p className="-mt-2 mb-4 text-sm text-emerald-700">{upload.ok}</p>}
        {upload.messages.length > 0 && (
          <ul className="-mt-2 mb-4 list-disc space-y-0.5 rounded-lg bg-red-50 py-2 pl-8 pr-3 text-sm text-red-700">
            {upload.messages.map((m) => <li key={m}>{m}</li>)}
          </ul>
        )}

        <div className="grid gap-4 sm:grid-cols-2">
          {METRIC_KEYS.map((key) => (
            <Field key={key} label={METRIC_LABELS[key]} error={errors[key]}>
              <Input inputMode="decimal" value={form.metrics[key]} invalid={!!errors[key]}
                className="text-right tabular-nums" onChange={(e) => setMetric(key, e.target.value)} placeholder="0.00" />
            </Field>
          ))}
        </div>

        {ratios && (
          <dl className="mt-5 grid grid-cols-2 gap-3 border-t border-line pt-4 sm:grid-cols-4">
            {ratios.map(([label, value]) => (
              <div key={label}>
                <dt className="text-xs text-muted">{label}</dt>
                <dd className="text-sm font-semibold tabular-nums">{value}</dd>
              </div>
            ))}
          </dl>
        )}
      </Card>
    </div>
  );
}
