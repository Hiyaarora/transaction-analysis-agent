/** The assessment questions carried inside the dataset file.
 *
 *  The list, its length and every question's text come from the API. Moving
 *  between questions is free: an answer already obtained for this dataset is
 *  shown from the cache, and only Ask reaches the backend.
 */

import { useEffect, useState } from "react";
import type { Session } from "../hooks/useSession";
import type { AskResponse } from "../types/api";
import { ResultView } from "./ResultView";
import { Button, Label, Panel, Spinner } from "./ui";

export function QuestionRunner({ session }: { session: Session }) {
  const questions = session.dataset?.questions ?? [];
  const [index, setIndex] = useState(0);
  const [result, setResult] = useState<AskResponse | null>(null);
  const [pending, setPending] = useState(false);

  const question = questions[index];

  // A new dataset means a new list and no carried-over answers.
  useEffect(() => {
    setIndex(0);
    setResult(null);
  }, [session.datasetToken]);

  // Showing a question shows whatever was already answered for it - no request.
  useEffect(() => {
    setResult(question ? session.cachedAnswer(question) : null);
  }, [question, session]);

  if (questions.length === 0) return null;

  async function ask() {
    setPending(true);
    try {
      setResult(await session.ask(question));
    } finally {
      setPending(false);
    }
  }

  return (
    <Panel label="Assessment questions" className="p-5">
      <div className="flex items-baseline justify-between gap-4">
        <Label>Assessment questions</Label>
        <p className="label">
          Question {index + 1} of {questions.length}
        </p>
      </div>

      <p className="mt-4 text-lg leading-snug font-medium">{question}</p>

      <div className="mt-5 flex flex-wrap items-center gap-3">
        <Button variant="primary" onClick={ask} disabled={pending}>
          {pending ? <Spinner label="Analysing..." /> : "Ask question"}
        </Button>
        {pending && (
          <p className="font-mono text-xs text-faint">Planning &rarr; Validating &rarr; Computing</p>
        )}
      </div>

      {result && (
        <div className="mt-5">
          <ResultView result={result} />
        </div>
      )}

      <nav className="mt-5 flex items-center justify-between border-t border-line pt-4">
        <Button onClick={() => setIndex((i) => i - 1)} disabled={index === 0}>
          &larr; Previous
        </Button>
        <Button onClick={() => setIndex((i) => i + 1)} disabled={index >= questions.length - 1}>
          Next &rarr;
        </Button>
      </nav>
    </Panel>
  );
}
