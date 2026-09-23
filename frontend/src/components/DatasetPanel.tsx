/** Left column: choose a dataset.
 *
 *  The loaded dataset's profile is not shown here. The dataset it was computed
 *  from is named in each result's explanation, which is where it matters.
 */

import { useRef } from "react";
import type { Session } from "../hooks/useSession";
import { Button, Label, Notice, Panel, Spinner } from "./ui";

export function DatasetPanel({ session }: { session: Session }) {
  const fileInput = useRef<HTMLInputElement>(null);
  const { datasetStatus, datasetError, loadAssessment, upload } = session;
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
    </div>
  );
}
