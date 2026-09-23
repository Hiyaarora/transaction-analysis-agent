/** Free-text questions.
 *
 *  The text is sent verbatim to the same endpoint the assessment questions
 *  use. There is no inspection of what was typed: no keyword matching, no
 *  routing, no special cases. Whatever comes back is rendered by the same
 *  ResultView.
 */

import { useEffect, useState } from "react";
import type { Session } from "../hooks/useSession";
import { useSlowRequest } from "../hooks/useSlowRequest";
import type { AskResponse } from "../types/api";
import { ResultView } from "./ResultView";
import { Button, Label, Panel, Spinner, WakingNotice } from "./ui";

export function CustomQuestion({ session }: { session: Session }) {
  const [text, setText] = useState("");
  const [result, setResult] = useState<AskResponse | null>(null);
  const [pending, setPending] = useState(false);
  const slow = useSlowRequest(pending);

  const question = text.trim();

  // An answer belongs to the dataset it was computed from. When the active
  // dataset is replaced, leaving it on screen would show one file's numbers
  // beside another file's profile.
  useEffect(() => {
    setResult(null);
  }, [session.datasetToken]);

  async function ask() {
    if (!question) return;
    // Drop the previous answer before the new one is requested: leaving it on
    // screen during the wait invites reading it as the answer to this question.
    setResult(null);
    setPending(true);
    try {
      setResult(await session.ask(question));
    } finally {
      setPending(false);
    }
  }

  return (
    <Panel label="Ask your own question" className="p-5">
      <Label>Ask your own question</Label>
      <div className="mt-3 flex flex-col gap-3 sm:flex-row">
        <input
          type="text"
          value={text}
          aria-label="Ask your own question"
          placeholder="What would you like to know about this dataset?"
          onChange={(event) => setText(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter" && !pending) void ask();
          }}
          className="min-w-0 flex-1 rounded-md border border-line-strong bg-canvas px-3 py-2 text-sm
                     placeholder:text-faint focus-visible:border-accent-cool focus-visible:outline-none"
        />
        <Button variant="primary" onClick={ask} disabled={pending || !question} className="sm:w-32">
          {pending ? <Spinner /> : "Ask"}
        </Button>
      </div>

      {pending && <p className="mt-3 font-mono text-xs text-faint">Planning &rarr; Validating &rarr; Computing</p>}
      {slow && <WakingNotice />}

      {result && session.dataset && (
        <div className="mt-5">
          <ResultView result={result} dataset={session.dataset} />
        </div>
      )}
    </Panel>
  );
}
