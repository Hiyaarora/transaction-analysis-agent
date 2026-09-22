/**
 * Step records -> readable sentences.
 *
 * This is presentation only: every number in a sentence was recorded by the
 * executor (rows in, rows out, groups in, groups out). Nothing here computes
 * anything, and nothing here knows what any particular question was about.
 *
 * The CLI has its own copy of this phrasing in renderer.py. The two are
 * deliberately separate: the terminal wants text, the browser wants
 * components, and neither should parse the other's output.
 */

import type { StepRecord } from "../types/api";

const OP_SYMBOL: Record<string, string> = {
  eq: "=",
  neq: "!=",
  gt: ">",
  gte: ">=",
  lt: "<",
  lte: "<=",
};

const OP_WORD: Record<string, string> = {
  eq: "equal to",
  neq: "not equal to",
  gt: "greater than",
  gte: "at least",
  lt: "less than",
  lte: "at most",
};

const AGG_VERB: Record<string, string> = {
  sum: "Summed",
  mean: "Averaged",
  median: "Took the median of",
  min: "Took the minimum of",
  max: "Took the maximum of",
};

/** Formulas for derived metrics, matching app/contract.py. */
const METRIC_FORMULAS: Record<string, string> = {
  revenue: "units * unit_price * (1 - discount)",
};

export function describeStep(step: StepRecord): string {
  const args = step.args as Record<string, never>;

  switch (step.tool) {
    case "filter_rows":
      return `Filtered transactions where ${condition(args)} (${step.rows_out} of ${step.rows_in} rows kept).`;

    case "compute_metric": {
      const metric = String(args.metric);
      const formula = METRIC_FORMULAS[metric] ?? "a fixed formula";
      return `Computed ${metric} for each of the ${rows(step.rows_in)} (${metric} = ${formula}).`;
    }

    case "aggregate":
      return args.func === "count"
        ? `Counted the ${rows(step.rows_in)}.`
        : `${AGG_VERB[String(args.func)]} ${args.column} over ${rows(step.rows_in)}.`;

    case "group_by": {
      const body =
        args.func === "count"
          ? "counted the transactions in each group"
          : `${AGG_VERB[String(args.func)].toLowerCase()} ${args.column} within each group`;
      return `Grouped ${step.rows_in} transactions by ${args.by} and ${body} (${step.groups_out} groups).`;
    }

    case "filter_groups":
      return `Kept groups whose value is ${groupCondition(args)} (${step.groups_out} of ${step.groups_in} groups kept).`;

    case "select_extreme":
      return `Selected the group with the ${args.mode} value (${step.groups_in} groups compared).`;

    default:
      return step.tool;
  }
}

function rows(count: number | null): string {
  return `${count} matching transaction${count === 1 ? "" : "s"}`;
}

function condition(args: Record<string, never>): string {
  const column = String(args.column);
  const op = String(args.op);
  const value = args.value as unknown;

  if (op === "is_null") return `${column} is missing`;
  if (op === "not_null") return `${column} is present`;
  if (op === "in" && Array.isArray(value)) return `${column} is one of ${value.map(plain).join(", ")}`;
  if (op === "between" && Array.isArray(value)) return `${column} between ${plain(value[0])} and ${plain(value[1])}`;
  return `${column} ${OP_SYMBOL[op] ?? op} ${plain(value)}`;
}

function groupCondition(args: Record<string, never>): string {
  const op = String(args.op);
  const value = args.value as unknown;
  if (op === "between" && Array.isArray(value)) {
    return `between ${money(value[0])} and ${money(value[1])}`;
  }
  return `${OP_WORD[op] ?? op} ${money(value)}`;
}

/** A filter value as the user would recognise it, without forcing decimals. */
function plain(value: unknown): string {
  return String(value);
}

function money(value: unknown): string {
  const numeric = Number(value);
  return Number.isFinite(numeric)
    ? numeric.toLocaleString("en-GB", { minimumFractionDigits: 2, maximumFractionDigits: 2 })
    : String(value);
}
