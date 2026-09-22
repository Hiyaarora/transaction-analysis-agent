/** One AskResponse, as three sections or as a status.
 *
 *  Answer            what was asked, and the value Python computed
 *  Operations        one line per executed step, built from its record
 *  Explanation       where the number came from, and what was left out
 *
 *  Every status has a branch, and nothing here turns a non-answer into a
 *  number. The only judgement the UI makes is formatting.
 */

import type { AskResponse, DatasetState, Execution, GroupsExecution } from "../types/api";
import { isAnswer } from "../types/api";
import { describeStep } from "./operations";
import { Label } from "./ui";

type Answer = Extract<AskResponse, { intent: string }>;

export function ResultView({ result, dataset }: { result: AskResponse; dataset: DatasetState }) {
  if (!isAnswer(result)) return <StatusCard status={result.status} message={result.message} />;

  return (
    <section aria-label="Result" className="flex flex-col gap-4">
      <AnswerCard result={result} />
      <Operations execution={result.execution} />
      <Explanation result={result} dataset={dataset} />
    </section>
  );
}

function AnswerCard({ result }: { result: Answer }) {
  const noData = result.status === "no_data";
  return (
    <div className="rounded-lg border border-line-strong bg-raised p-5">
      <Label>Answer</Label>
      <p className="mt-2 text-sm text-muted">{result.intent}</p>
      <div className="mt-3">
        {noData ? (
          <p className="text-lg font-medium text-warning">
            No data &mdash; no transactions matched, so there is nothing to report.
          </p>
        ) : (
          <AnswerValue execution={result.execution} />
        )}
      </div>
    </div>
  );
}

function AnswerValue({ execution }: { execution: Execution }) {
  if (execution.kind === "scalar") {
    return <p className="text-4xl font-semibold tracking-tight tabular-nums">{formatNumber(execution.value)}</p>;
  }

  if (execution.kind === "groups") {
    return (
      <dl className="flex flex-col">
        {execution.rows.map((row) => (
          <div
            key={row.group}
            className="flex items-baseline justify-between gap-6 border-b border-line py-2 last:border-0"
          >
            <dt className="font-mono text-sm">{row.group}</dt>
            <dd className="flex items-baseline gap-3">
              {row.rows_used !== row.rows_total && (
                <span className="font-mono text-xs text-faint">
                  {row.rows_used} of {row.rows_total} rows
                </span>
              )}
              <span className="text-xl font-semibold tabular-nums">
                {row.value === null ? <span className="text-sm text-muted">no value</span> : formatNumber(row.value)}
              </span>
            </dd>
          </div>
        ))}
      </dl>
    );
  }

  return (
    <div className="flex flex-wrap items-baseline gap-x-4 gap-y-1">
      <span className="font-mono text-2xl font-semibold">{execution.groups.join(", ")}</span>
      <span className="text-xl tabular-nums text-muted">{formatNumber(execution.value)}</span>
      {execution.groups.length > 1 && <span className="text-xs text-warning">tied</span>}
    </div>
  );
}

function Operations({ execution }: { execution: Execution }) {
  return (
    <div className="rounded-lg border border-line p-5">
      <Label>Operations performed</Label>
      <ol className="mt-3 flex list-none flex-col gap-2">
        {execution.steps.map((step, position) => (
          <li key={position} className="flex gap-3 text-sm leading-relaxed">
            <span className="font-mono text-xs tabular-nums text-faint">{position + 1}.</span>
            <span className="text-ink">{describeStep(step)}</span>
          </li>
        ))}
      </ol>
    </div>
  );
}

function Explanation({ result, dataset }: { result: Answer; dataset: DatasetState }) {
  const notes: string[] = [];
  const execution = result.execution;

  if (result.status === "no_data") {
    notes.push(
      dataset.date_range
        ? `The dataset holds ${dataset.row_count} transactions spanning ${dataset.date_range.start} to ${dataset.date_range.end}.`
        : `The dataset holds ${dataset.row_count} transactions.`,
    );
  } else if (execution.kind === "scalar" && execution.rows_used !== execution.rows_total) {
    const skipped = execution.rows_total - execution.rows_used;
    notes.push(
      `Computed from ${execution.rows_used} of the ${execution.rows_total} matching transactions; ` +
        `${skipped} had a missing value and ${skipped === 1 ? "was" : "were"} left out rather than treated as zero.`,
    );
  }

  if (execution.kind === "groups") notes.push(...groupNotes(execution));

  return (
    <div className="rounded-lg border border-line p-5">
      <Label>Explanation</Label>
      <p className="mt-3 text-sm leading-relaxed text-muted">
        The result was computed deterministically in Python from the active dataset (
        <span className="font-mono text-ink">{dataset.source_name}</span>); the language model only chose which
        operations to run.
      </p>
      {notes.map((note) => (
        <p key={note} className="mt-2 text-sm leading-relaxed text-warning">
          {note}
        </p>
      ))}
    </div>
  );
}

function groupNotes(execution: GroupsExecution): string[] {
  const notes: string[] = [];
  const short = execution.rows.filter((row) => row.rows_used !== row.rows_total);
  if (short.length > 0) {
    const detail = short.map((row) => `${row.group}: ${row.rows_used} of ${row.rows_total}`).join("; ");
    notes.push(`Some groups had missing values and were computed from fewer rows (${detail}).`);
  }
  if (execution.rows_without_group > 0) {
    const count = execution.rows_without_group;
    notes.push(
      `${count} transaction${count === 1 ? "" : "s"} had no group value and ${
        count === 1 ? "was" : "were"
      } excluded from the grouping.`,
    );
  }
  if (execution.groups_without_value > 0) {
    const count = execution.groups_without_value;
    notes.push(
      `${count} group${count === 1 ? "" : "s"} had no computable value and ${
        count === 1 ? "was" : "were"
      } not compared.`,
    );
  }
  return notes;
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
    <section aria-label="Result" className={`rounded-lg border p-5 ${STATUS_STYLES[status]}`}>
      <p className="text-sm font-semibold">{STATUS_TITLES[status]}</p>
      <p className="mt-2 text-sm leading-relaxed text-muted">{message}</p>
    </section>
  );
}
