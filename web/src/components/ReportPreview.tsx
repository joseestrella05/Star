import type { ReportResponse, Section } from "../api";
import { Badge, Button, Card } from "./ui";

function tableRows(markdown: string) {
  return markdown
    .split("\n")
    .slice(2) // header row and |---| separator
    .map((line) => line.split("|").slice(1, -1).map((c) => c.trim()));
}

function tableHead(markdown: string) {
  const row = markdown.split("\n")[0];
  return row.split("|").slice(1, -1).map((c) => c.trim());
}

export function exportMarkdown(sections: Section[], commentary: string) {
  return sections.map((s) => (s.key === "commentary" ? commentary.trim() : s.markdown)).join("\n\n") + "\n";
}

export function footerText(report: ReportResponse) {
  return `Reporte ${report.report_id.slice(0, 8)} · modelo ${report.model.checkpoint_sha256.slice(0, 12)} · ${report.created_at.slice(0, 10)}`;
}

export function ReportPreview({
  report,
  commentary,
  onCommentary,
  onExport,
  exporting,
}: {
  report: ReportResponse;
  commentary: string;
  onCommentary: (text: string) => void;
  onExport: (format: "pdf" | "md") => void;
  exporting: "pdf" | "md" | null;
}) {
  const section = (key: Section["key"]) => report.sections.find((s) => s.key === key);
  const commentarySec = section("commentary");
  const opinionSec = section("opinion");
  const criteriaSec = section("criteria");
  const rows = tableRows(section("table")!.markdown);
  const edited = !!commentarySec && commentary.trim() !== commentarySec.markdown;

  return (
    <Card
      title="Vista previa del informe"
      aside={
        <div className="flex gap-2">
          <Button variant="secondary" disabled={!!exporting} onClick={() => onExport("md")}>
            {exporting === "md" ? "Exportando…" : "Markdown"}
          </Button>
          <Button disabled={!!exporting || !commentary.trim()} onClick={() => onExport("pdf")}>
            {exporting === "pdf" ? "Generando PDF…" : "Descargar PDF"}
          </Button>
        </div>
      }
    >
      <article className="space-y-5">
        <h3 className="border-b-2 border-ink pb-2 text-lg font-bold">{section("header")?.markdown.replace(/^#\s*/, "") ?? ""}</h3>

        <table className="w-full text-sm">
          <tbody>
            {rows.map(([label, value]) => (
              <tr key={label} className="border-b border-line">
                <td className="py-1.5 text-muted">{label}</td>
                <td className="py-1.5 text-right font-medium tabular-nums">{value}</td>
              </tr>
            ))}
          </tbody>
        </table>

        <div>
          <div className="mb-1.5 flex items-center gap-2">
            <span className="text-xs font-medium text-muted">Análisis</span>
            <Badge tone="slate">Calculado · no editable</Badge>
          </div>
          <p className="text-sm leading-relaxed">{section("analysis")?.markdown ?? ""}</p>
        </div>

        {criteriaSec && (
          <div>
            <div className="mb-1.5 flex items-center gap-2">
              <span className="text-xs font-medium text-muted">Criterios</span>
              <Badge tone="slate">Calculado · no editable</Badge>
            </div>
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b-2 border-line">
                  {tableHead(criteriaSec.markdown).map((h) => (
                    <th key={h} className="py-1.5 pr-3 text-left text-xs font-medium text-muted first:pl-0 last:pr-0">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {tableRows(criteriaSec.markdown).map((cols, i) => (
                  <tr key={i} className="border-b border-line">
                    {cols.map((c, j) => (
                      <td key={j} className={`py-1.5 pr-3 first:pl-0 last:pr-0 ${j === 0 ? "font-medium" : "tabular-nums"} ${c === "No" || c === "No aplica" ? "text-red-600" : c === "Sí" || c === "Yes" ? "text-emerald-700" : ""}`}>{c}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {commentarySec && (
          <div>
            <div className="mb-1.5 flex flex-wrap items-center gap-2">
              <span className="text-xs font-medium text-muted">Comentario</span>
              {/* A manual edit voids the verification: say so instead of keeping the green badge. */}
              {edited ? (
                <Badge tone="amber">Editado manualmente · cifras sin verificar</Badge>
              ) : commentarySec.source === "model" ? (
                <Badge tone="green">Redactado por Star · cifras verificadas</Badge>
              ) : (
                <Badge tone="amber">Texto estándar</Badge>
              )}
            </div>
            <textarea
              value={commentary}
              onChange={(e) => onCommentary(e.target.value)}
              rows={3}
              className="w-full resize-y rounded-lg border border-line p-3 text-sm leading-relaxed outline-none focus:border-brand focus:ring-2 focus:ring-brand/20"
            />
          </div>
        )}

        {opinionSec && (
          <div>
            <div className="mb-1.5 flex items-center gap-2">
              <span className="text-xs font-medium text-muted">Dictamen</span>
              <Badge tone="slate">Determinista · no editable</Badge>
            </div>
            <p className="text-sm leading-relaxed whitespace-pre-line">{opinionSec.markdown}</p>
          </div>
        )}

        <footer className="space-y-1 border-t border-line pt-3 text-xs text-muted">
          <p>{footerText(report)}</p>
          <p>
            Verificación: {report.verification.attempts} intento(s)
            {report.verification.rejected_figures.length > 0 && ` · cifras descartadas: ${report.verification.rejected_figures.join(", ")}`}
            {report.verification.rejected_commentary.length > 0 &&
              ` · ${report.verification.rejected_commentary.length} comentario(s) descartado(s) por contradecir las métricas`}
          </p>
        </footer>
      </article>
    </Card>
  );
}
