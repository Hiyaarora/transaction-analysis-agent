/** Renders one AskResponse.
 *
 *  Every status has a branch: an answer is shown as an answer, and a
 *  clarification, rejection or failure is shown as itself. Nothing here turns
 *  a non-answer into an answer.
 *
 *  Phase 12E expands the success case into the full three-section display.
 */

import type { AskResponse, Execution } from "../types/api";
import { isAnswer } from "../types/api";
import { Label } from "./ui";

export function ResultView({ result }: { result: AskResponse }) {
  if (!isAnswer(result)) return <StatusCard status={result.status} message={result.message} />;

  const noData = result.status === "no_data";
  return (
    <div className="rounded-lg border border-line bg-raised p-5">
      <Label>Answer</Label>
      <p className="mt-2 text-sm text-muted">{result.intent}</p>
      <div className="mt-3">
        {noData ? (
          <p className="text-base text-warning">No data. No transactions matched.</p>
        ) : (
          <AnswerValue execution={result.execution} />
        )}
      </div>
    </div>
  );
}

function AnswerValue({ execution }: { execution: Execution }) {
  if (execution.kind === "scalar") {
    return <p className="text-3xl font-semibold tabular-nums">{formatNumber(execution.value)}</p>;
  }

  if (execution.kind === "groups") {
    return (
      <dl className="flex flex-col gap-1.5">
        {execution.rows.map((row) => (
          <div key={row.group} className="flex items-baseline justify-between gap-6 border-b border-line pb-1.5">
            <dt className="font-mono text-sm">{row.group}</dt>
            <dd className="text-lg font-semibold tabular-nums">
              {row.value === null ? <span className="text-sm text-muted">no value</span> : formatNumber(row.value)}
            </dd>
          </div>
        ))}
      </dl>
    );
  }

  return (
    <p className="flex items-baseline gap-3">
      <span className="font-mono text-xl font-semibold">{execution.groups.join(", ")}</span>
      <span className="text-lg tabular-nums text-muted">{formatNumber(execution.value)}</span>
      {execution.groups.length > 1 && <span className="text-xs text-muted">(tied)</span>}
    </p>
  );
}

/** Counts stay whole; everything else gets two decimals. The value itself was
 *  computed in Python - this only chooses how to print it. */
export function formatNumber(value: number | null): string {
  if (value === null) return "-";
  return Number.isInteger(value)
    ? value.toLocaleString("en-GB")
    : value.toLocaleString("en-GB", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

const STATUS_TITLES = {
  clarification_required: "Clarification required",
  rejected: "Request rejected",
  error: "Something went wrong",
} as const;

const STATUS_STYLES = {
  clarification_required: "border-warning/40 bg-warning/10",
  rejected: "border-line-strong bg-raised",
  error: "border-danger/40 bg-danger/10",
} as const;

function StatusCard({
  status,
  message,
}: {
  status: "clarification_required" | "rejected" | "error";
  message: string;
}) {
  return (
    <div className={`rounded-lg border p-5 ${STATUS_STYLES[status]}`}>
      <p className="text-sm font-semibold">{STATUS_TITLES[status]}</p>
      <p className="mt-2 text-sm leading-relaxed text-muted">{message}</p>
    </div>
  );
}
