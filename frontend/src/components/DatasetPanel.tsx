/** Left column: choose a dataset.
 *
 *  Each of the two ways in is a control that becomes its result. Once the
 *  assessment dataset is loaded, its button is the file; once a CSV has been
 *  uploaded, the upload button is that file. The other way in keeps its button,
 *  so switching is always one click away, and nothing restates elsewhere what
 *  is already loaded.
 */

import { useRef, useState } from "react";
import { ApiError, downloadDataset } from "../api/client";
import { useSlowRequest } from "../hooks/useSlowRequest";
import type { Session } from "../hooks/useSession";
import { Button, Label, Notice, Panel, Spinner, WakingNotice } from "./ui";

export function DatasetPanel({ session }: { session: Session }) {
  const fileInput = useRef<HTMLInputElement>(null);
  const [downloadError, setDownloadError] = useState<string | null>(null);
  const { dataset, datasetStatus, datasetError, datasetSource, loadAssessment, upload, sessionId } = session;
  const busy = datasetStatus === "loading";
  const slow = useSlowRequest(busy);

  async function download() {
    if (!dataset) return;
    setDownloadError(null);
    try {
      await downloadDataset(sessionId, dataset.source_name);
    } catch (error) {
      setDownloadError(error instanceof ApiError ? error.message : "The dataset could not be downloaded.");
    }
  }

  const loadedFile = dataset && (
    <LoadedFile name={dataset.source_name} rows={dataset.row_count} onDownload={download} />
  );

  return (
    <div className="flex flex-col gap-4">
      <Panel className="p-5">
        <Label>Dataset</Label>
        <h2 className="mt-3 text-base font-semibold">Assessment dataset</h2>
        {/* The name is only a caption until the file is loaded; after that the
            control below carries it, and repeating it would say it twice. */}
        {datasetSource !== "assessment" && (
          <p className="mt-1 font-mono text-xs text-muted">project_4.csv</p>
        )}
        <p className="mt-3 text-sm leading-relaxed text-muted">
          The dataset supplied with the exercise, including the questions embedded in the file.
        </p>

        <div className="mt-4">
          {datasetSource === "assessment" ? (
            loadedFile
          ) : (
            <Button variant="primary" className="w-full" onClick={loadAssessment} disabled={busy}>
              {busy ? <Spinner /> : "Use assessment dataset"}
            </Button>
          )}
        </div>

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

        {datasetSource === "upload" ? (
          <>
            {loadedFile}
            <button
              onClick={() => fileInput.current?.click()}
              disabled={busy}
              className="mt-3 text-xs text-muted underline-offset-2 transition-colors duration-150
                         hover:text-ink hover:underline disabled:opacity-40"
            >
              Upload a different CSV
            </button>
          </>
        ) : (
          <>
            <Button className="w-full" onClick={() => fileInput.current?.click()} disabled={busy}>
              Upload CSV
            </Button>
            <p className="mt-3 text-xs leading-relaxed text-faint">
              A transaction CSV with id, date, region, product, units, unit_price and discount. Up to 5 MB.
            </p>
          </>
        )}
        {slow && <WakingNotice />}
      </Panel>

      {datasetError && <Notice tone="error">{datasetError}</Notice>}
      {downloadError && <Notice tone="error">{downloadError}</Notice>}
    </div>
  );
}

/** The loaded file, as one control: the whole tile downloads it.
 *
 *  A single button rather than a row with a button inside it - nesting one
 *  button in another is invalid, and it would make the obvious target (the
 *  name) the one part that did nothing.
 */
function LoadedFile({ name, rows, onDownload }: { name: string; rows: number; onDownload: () => void }) {
  return (
    <button
      onClick={onDownload}
      aria-label={`Download ${name}`}
      title="Download this CSV"
      className="group flex w-full items-center gap-3 rounded-md border border-line-strong bg-raised
                 px-3 py-2.5 text-left transition-colors duration-150 hover:border-faint
                 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent-cool"
    >
      <CheckIcon />
      <span className="min-w-0 flex-1">
        <span className="block truncate font-mono text-sm text-ink">{name}</span>
        <span className="block text-xs text-muted">
          {rows} transaction{rows === 1 ? "" : "s"} loaded
        </span>
      </span>
      <span className="shrink-0 text-muted transition-colors duration-150 group-hover:text-ink">
        <DownloadIcon />
      </span>
    </button>
  );
}

function CheckIcon() {
  return (
    <svg viewBox="0 0 16 16" aria-hidden="true" className="size-4 shrink-0 text-positive" fill="none"
         stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <path d="m3.5 8.5 3 3 6-7" />
    </svg>
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
