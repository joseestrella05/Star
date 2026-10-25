export type Lang = "es" | "en";
export type Source = "computed" | "model" | "fallback";

export const METRIC_KEYS = [
  "revenue",
  "ebitda",
  "operating_cash_flow",
  "capex",
  "working_capital_change",
  "cash",
  "net_debt",
] as const;
export type MetricKey = (typeof METRIC_KEYS)[number];
export type Metrics = Record<MetricKey, string>;

export interface ReportRequest {
  company_name: string;
  period: string;
  language: Lang;
  metrics: Metrics;
  generation?: { temperature?: number; max_attempts?: number; seed?: number | null };
}

export interface Section {
  key: "header" | "table" | "analysis" | "commentary";
  markdown: string;
  source: Source;
}

export interface ReportResponse {
  report_id: string;
  created_at: string;
  markdown: string;
  sections: Section[];
  verification: { passed: boolean; attempts: number; rejected_figures: string[] };
  model: { checkpoint_sha256: string; params: number };
}

export interface Health {
  status: string;
  device: string;
  params: number;
  checkpoint_sha256: string;
}

export class ApiError extends Error {
  constructor(public messages: string[]) {
    super(messages.join("\n"));
  }
}

/** FastAPI errors: validation gives [{loc, msg}], our handlers give strings or string lists. */
function readDetail(detail: unknown): string[] {
  if (typeof detail === "string") return [detail];
  if (Array.isArray(detail)) {
    return detail.map((d) =>
      typeof d === "string" ? d : `${(d.loc ?? []).filter((p: unknown) => p !== "body").join(".")}: ${d.msg}`,
    );
  }
  return ["Error inesperado del servidor"];
}

async function request<T>(url: string, init?: RequestInit, as: "json" | "blob" = "json"): Promise<T> {
  const res = await fetch(url, init);
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }));
    throw new ApiError(readDetail(body.detail));
  }
  return (as === "json" ? res.json() : res.blob()) as Promise<T>;
}

const jsonInit = (body: unknown): RequestInit => ({
  method: "POST",
  headers: { "content-type": "application/json" },
  body: JSON.stringify(body),
});

export const api = {
  health: () => request<Health>("/health"),
  generate: (req: ReportRequest) => request<ReportResponse>("/api/v1/reports/generate", jsonInit(req)),
  ingest: (file: File) =>
    request<{ fields: Partial<Pick<ReportRequest, "company_name" | "period" | "language">>; metrics: Record<string, string> }>(
      "/api/v1/reports/ingest",
      { method: "POST", body: file },
    ),
  export: (markdown: string, format: "md" | "pdf", footer: string) =>
    request<Blob>("/api/v1/reports/export", jsonInit({ markdown, format, footer }), "blob"),
  templateUrl: "/api/v1/reports/template.csv",
};
