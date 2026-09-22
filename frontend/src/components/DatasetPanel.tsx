/** Left column: choose a dataset, then see what was actually loaded.
 *
 *  Every value shown comes from the API's DataProfile. Nothing about the
 *  data - regions, products, counts, dates - is written in this file.
 */

import { useRef } from "react";
import type { Session } from "../hooks/useSession";
import type { DatasetState } from "../types/api";
import { Button, Label, Notice, Panel, Spinner } from "./ui";

export function DatasetPanel({ session }: { session: Session }) {
  const fileInput = useRef<HTMLInputElement>(null);
  const { dataset, datasetStatus, datasetError, loadAssessment, upload } = session;
  const busy = datasetStatus === "loading";

  return (
    <div className="flex flex-col gap-4">
      <Panel className="p-5">
        <Label>Dataset</Label>
        <h2 className="mt-3 text-base font-semibold">Assessment dataset</h2>
        <p className="mt-1 font-mono text-xs text-muted">project_4.csv</p>
        <p className="mt-3 text-sm leading-relaxed text-muted">
          The dataset supplied with the exercise, including the questions embedded in the file.
        </p>
        <Button variant="primary" className="mt-4 w-full" onClick={loadAssessment} disabled={busy}>
          {busy ? <Spinner /> : "Use assessment dataset"}
        </Button>

        <div className="my-5 flex items-center gap-3">
          <span className="h-px flex-1 bg-line" />
          <span className="label">or</span>
          <span className="h-px flex-1 bg-line" />
        </div>

        <input
          ref={fileInput}
          type="file"
          accept=".csv,text/csv"
          className="sr-only"
          aria-label="Upload a CSV file"
          onChange={(event) => {
            const file = event.target.files?.[0];
            event.target.value = ""; // allow re-selecting the same file
            if (file) void upload(file);
          }}
        />
        <Button className="w-full" onClick={() => fileInput.current?.click()} disabled={busy}>
          Upload CSV
        </Button>
        <p className="mt-3 text-xs leading-relaxed text-faint">
          A transaction CSV with id, date, region, product, units, unit_price and discount. Up to 5 MB.
        </p>
      </Panel>

      {datasetError && <Notice tone="error">{datasetError}</Notice>}
      {dataset && <ActiveDataset dataset={dataset} />}
    </div>
  );
}

function ActiveDataset({ dataset }: { dataset: DatasetState }) {
  const missing = Object.entries(dataset.null_counts).filter(([, count]) => count > 0);
  const unreadable = Object.entries(dataset.parse_error_counts);

  return (
    <Panel label="Active dataset" className="p-5">
      <Label>Active dataset</Label>
      <p className="mt-3 truncate font-mono text-sm text-ink" title={dataset.source_name}>
        {dataset.source_name}
      </p>

      <div className="mt-4 flex flex-wrap gap-x-6 gap-y-2 border-y border-line py-3">
        <Stat value={dataset.row_count} unit={dataset.row_count === 1 ? "transaction" : "transactions"} />
        {Object.entries(dataset.categorical_values).map(([column, values]) => (
          <Stat key={column} value={values.length} unit={values.length === 1 ? column : `${column}s`} />
        ))}
      </div>

      {dataset.date_range && (
        <p className="mt-3 font-mono text-xs text-muted">
          {dataset.date_range.start} &rarr; {dataset.date_range.end}
        </p>
      )}

      <dl className="mt-4 flex flex-col gap-3">
        {Object.entries(dataset.categorical_values).map(([column, values]) => (
          <div key={column}>
            <dt className="label">{column}</dt>
            <dd className="mt-1 flex flex-wrap gap-1.5">
              {values.map((value) => (
                <span
                  key={value}
                  className="rounded border border-line bg-raised px-2 py-0.5 font-mono text-xs text-ink"
                >
                  {value}
                </span>
              ))}
            </dd>
          </div>
        ))}
      </dl>

      <dl className="mt-4 flex flex-col gap-2 border-t border-line pt-3 text-xs">
        <Row term="Metrics" detail={dataset.supported_metrics.join(", ")} />
        <Row term="Missing" detail={missing.length ? missing.map(([c, n]) => `${c} (${n})`).join(", ") : "none"} />
        {unreadable.length > 0 && (
          <Row term="Unreadable" detail={unreadable.map(([c, n]) => `${c} (${n})`).join(", ")} />
        )}
        {dataset.extra_columns.length > 0 && (
          <Row term="Not analysable" detail={dataset.extra_columns.join(", ")} />
        )}
      </dl>
    </Panel>
  );
}

function Stat({ value, unit }: { value: number; unit: string }) {
  return (
    <p className="flex items-baseline gap-1.5">
      <span className="text-xl font-semibold tabular-nums">{value}</span>
      <span className="text-xs text-muted">{unit}</span>
    </p>
  );
}

function Row({ term, detail }: { term: string; detail: string }) {
  return (
    <div className="flex gap-3">
      <dt className="label w-24 shrink-0">{term}</dt>
      <dd className="font-mono text-xs text-muted">{detail}</dd>
    </div>
  );
}
