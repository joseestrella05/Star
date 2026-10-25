import { useEffect, useState } from "react";
import { api, ApiError, type Health, type ReportResponse } from "./api";
import { MetricsForm, normalizeMetrics, validate, type FormState } from "./components/MetricsForm";
import { exportMarkdown, footerText, ReportPreview } from "./components/ReportPreview";
import { Badge, Button } from "./components/ui";

const EMPTY: FormState = {
  company_name: "",
  period: "Q2 2026",
  language: "es",
  metrics: { revenue: "", ebitda: "", operating_cash_flow: "", capex: "", working_capital_change: "0", cash: "", net_debt: "" },
};

function download(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const a = Object.assign(document.createElement("a"), { href: url, download: filename });
  a.click();
  URL.revokeObjectURL(url);
}

export default function App() {
  const [health, setHealth] = useState<Health | null>(null);
  const [form, setForm] = useState<FormState>(EMPTY);
  const [touched, setTouched] = useState(false);
  const [report, setReport] = useState<ReportResponse | null>(null);
  const [commentary, setCommentary] = useState("");
  const [busy, setBusy] = useState(false);
  const [exporting, setExporting] = useState<"pdf" | "md" | null>(null);
  const [errors, setErrors] = useState<string[]>([]);

  useEffect(() => {
    api.health().then(setHealth).catch(() => setHealth(null));
  }, []);

  const fieldErrors = touched ? validate(form) : {};

  async function generate() {
    setTouched(true);
    if (Object.keys(validate(form)).length) return;
    setBusy(true);
    setErrors([]);
    try {
      const r = await api.generate({ ...form, metrics: normalizeMetrics(form.metrics) });
      setReport(r);
      setCommentary(r.sections.find((s) => s.key === "commentary")!.markdown);
    } catch (e) {
      setErrors(e instanceof ApiError ? e.messages : ["No se pudo conectar con el servidor"]);
    } finally {
      setBusy(false);
    }
  }

  async function exportReport(format: "pdf" | "md") {
    if (!report) return;
    setExporting(format);
    setErrors([]);
    try {
      const blob = await api.export(exportMarkdown(report.sections, commentary), format, footerText(report));
      const slug = report.sections[0].markdown.replace(/^#\s*/, "").replace(/[^\p{L}\p{N}]+/gu, "-").toLowerCase();
      download(blob, `${slug}.${format}`);
    } catch (e) {
      setErrors(e instanceof ApiError ? e.messages : ["No se pudo exportar el informe"]);
    } finally {
      setExporting(null);
    }
  }

  return (
    <div className="min-h-screen">
      <header className="border-b border-line bg-white">
        <div className="mx-auto flex max-w-6xl flex-wrap items-center justify-between gap-3 px-4 py-3 sm:px-6">
          <div className="flex items-center gap-3">
            <span className="grid size-8 place-items-center rounded-lg bg-brand text-sm font-bold text-white">S</span>
            <div>
              <h1 className="text-sm font-semibold leading-tight">Star</h1>
              <p className="text-xs text-muted">Análisis de flujo de caja · on-premise</p>
            </div>
          </div>
          {health ? (
            <div className="flex flex-wrap items-center gap-2">
              <Badge tone="green">Servidor local · sin conexión externa</Badge>
              <span className="font-mono text-xs text-muted" title={health.checkpoint_sha256}>
                modelo {health.checkpoint_sha256.slice(0, 12)} · {(health.params / 1e6).toFixed(2)}M · {health.device}
              </span>
            </div>
          ) : (
            <Badge tone="amber">Servidor no disponible</Badge>
          )}
        </div>
      </header>

      <main className="mx-auto grid max-w-6xl gap-6 px-4 py-6 sm:px-6 lg:grid-cols-2">
        <div className="space-y-5">
          <MetricsForm form={form} onChange={setForm} errors={fieldErrors} />
          <Button className="w-full" onClick={generate} disabled={busy}>
            {busy ? "Generando informe…" : "Generar informe"}
          </Button>
        </div>

        <div className="space-y-4">
          {errors.length > 0 && (
            <ul role="alert" className="list-disc space-y-0.5 rounded-lg bg-red-50 py-2 pl-8 pr-3 text-sm text-red-700">
              {errors.map((m) => <li key={m}>{m}</li>)}
            </ul>
          )}
          {report ? (
            <ReportPreview report={report} commentary={commentary} onCommentary={setCommentary} onExport={exportReport} exporting={exporting} />
          ) : (
            <div className="grid h-full min-h-64 place-items-center rounded-xl border border-dashed border-slate-300 p-8 text-center text-sm text-muted">
              Complete las métricas y genere el informe. Las cifras se calculan en el servidor; el modelo solo redacta el comentario y
              cada cifra que escribe se verifica.
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
