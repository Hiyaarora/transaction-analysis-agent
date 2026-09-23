/** Left column: choose a dataset, and see which one is loaded.
 *
 *  The full profile is not shown here - the dataset an answer came from is
 *  named in that answer's explanation. What this does show is the name of the
 *  file in use and a way to download it, so it can be opened in whatever the
 *  person normally uses for a spreadsheet.
 */

import { useRef, useState } from "react";
import { ApiError, downloadDataset } from "../api/client";
import type { Session } from "../hooks/useSession";
import { Button, Label, Notice, Panel, Spinner } from "./ui";

export function DatasetPanel({ session }: { session: Session }) {
  const fileInput = useRef<HTMLInputElement>(null);
  const [downloadError, setDownloadError] = useState<string | null>(null);
  const { dataset, datasetStatus, datasetError, loadAssessment, upload, sessionId } = session;
  const busy = datasetStatus === "loading";

  async function download() {
    if (!dataset) return;
    setDownloadError(null);
    try {
      await downloadDataset(sessionId, dataset.source_name);
    } catch (error) {
      setDownloadError(error instanceof ApiError ? error.message : "The dataset could not be downloaded.");
    }
  }

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

        {dataset && (
          <div className="mt-5 flex items-center gap-3 border-t border-line pt-4">
            <div className="min-w-0 flex-1">
              <p className="label">In use</p>
              <p className="mt-1 truncate font-mono text-sm text-ink" title={dataset.source_name}>
                {dataset.source_name}
              </p>
            </div>
            <button
              onClick={download}
              aria-label={`Download ${dataset.source_name}`}
              title="Download this CSV"
              className="shrink-0 rounded-md border border-line-strong p-2 text-muted transition-colors
                         duration-150 hover:border-faint hover:text-ink focus-visible:outline-2
                         focus-visible:outline-offset-2 focus-visible:outline-accent-cool"
            >
              <DownloadIcon />
            </button>
          </div>
        )}
      </Panel>

      {datasetError && <Notice tone="error">{datasetError}</Notice>}
      {downloadError && <Notice tone="error">{downloadError}</Notice>}
    </div>
  );
}

function DownloadIcon() {
  return (
    <svg viewBox="0 0 16 16" aria-hidden="true" className="size-4" fill="none" stroke="currentColor"
         strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round">
      <path d="M8 2v8" />
      <path d="M4.5 7 8 10.5 11.5 7" />
      <path d="M2.5 13h11" />
    </svg>
  );
}
