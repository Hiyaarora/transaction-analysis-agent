/**
 * The wire contract, mirroring app/api/schemas.py.
 *
 * The unions are discriminated on purpose: TypeScript then forces every
 * caller to handle each status and each result kind, so a new backend state
 * cannot be silently ignored by the UI.
 */

export type DateRange = { start: string; end: string };

export type DatasetState = {
  source_name: string;
  row_count: number;
  columns: string[];
  categorical_values: Record<string, string[]>;
  numeric_ranges: Record<string, [number, number] | null>;
  date_range: DateRange | null;
  null_counts: Record<string, number>;
  parse_error_counts: Record<string, number>;
  extra_columns: string[];
  supported_metrics: string[];
  questions: string[];
};

export type StepRecord = {
  tool: string;
  args: Record<string, unknown>;
  rows_in: number | null;
  rows_out: number | null;
  groups_in: number | null;
  groups_out: number | null;
};

export type GroupRow = {
  group: string;
  value: number | null;
  rows_total: number;
  rows_used: number;
};

export type ScalarExecution = {
  kind: "scalar";
  steps: StepRecord[];
  value: number | null;
  rows_total: number;
  rows_used: number;
};

export type GroupsExecution = {
  kind: "groups";
  steps: StepRecord[];
  by: string;
  func: string;
  column: string | null;
  rows: GroupRow[];
  rows_without_group: number;
  groups_without_value: number;
};

export type ExtremeExecution = {
  kind: "extreme";
  steps: StepRecord[];
  mode: string;
  groups: string[];
  value: number | null;
};

export type Execution = ScalarExecution | GroupsExecution | ExtremeExecution;

export type AnswerResponse = {
  status: "success" | "no_data";
  question: string;
  intent: string;
  execution: Execution;
};

export type NonAnswerResponse = {
  status: "clarification_required" | "rejected" | "error";
  question: string;
  message: string;
};

export type AskResponse = AnswerResponse | NonAnswerResponse;

export type Health = { status: "ok"; llm_configured: boolean };

export function isAnswer(response: AskResponse): response is AnswerResponse {
  return response.status === "success" || response.status === "no_data";
}
