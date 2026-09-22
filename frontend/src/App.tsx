import { DatasetPanel } from "./components/DatasetPanel";
import { CustomQuestion } from "./components/CustomQuestion";
import { QuestionRunner } from "./components/QuestionRunner";
import { useSession } from "./hooks/useSession";
import { Panel } from "./components/ui";

export default function App() {
  const session = useSession();

  return (
    <div className="min-h-full">
      <header className="border-b border-line">
        <div className="mx-auto max-w-7xl px-6 py-8">
          <h1 className="text-2xl font-semibold tracking-tight sm:text-3xl">
            <span className="accent-text">Transaction Analysis</span> Agent
          </h1>
          <p className="mt-2 max-w-2xl text-sm leading-relaxed text-muted">
            Ask questions in plain English. The language model chooses the operations; Python performs
            every calculation on the loaded dataset.
          </p>
        </div>
      </header>

      <main className="mx-auto grid max-w-7xl gap-6 px-6 py-8 lg:grid-cols-[360px_minmax(0,1fr)]">
        <DatasetPanel session={session} />
        <div className="flex flex-col gap-6">
          {session.dataset ? (
            <>
              <QuestionRunner session={session} />
              <CustomQuestion session={session} />
            </>
          ) : (
            <EmptyState />
          )}
        </div>
      </main>
    </div>
  );
}

function EmptyState() {
  return (
    <Panel className="flex min-h-64 flex-col items-center justify-center gap-2 p-10 text-center">
      <p className="text-sm font-medium">No dataset loaded</p>
      <p className="max-w-sm text-sm leading-relaxed text-muted">
        Choose the assessment dataset or upload your own CSV to begin.
      </p>
    </Panel>
  );
}
