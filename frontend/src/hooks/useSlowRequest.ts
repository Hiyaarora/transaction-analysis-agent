/**
 * True once a request has been in flight long enough to need explaining.
 *
 * The deployed service sleeps after a period without traffic, and whichever
 * request wakes it waits for the whole cold start - tens of seconds. A
 * spinner alone is indistinguishable from a hang at that length, so the UI
 * says what the wait is for instead of leaving the user to guess.
 *
 * The threshold is a UI concern only: nothing is cancelled or retried here.
 * The request's own timeout still governs when it gives up.
 */

import { useEffect, useState } from "react";

/** Comfortably past a normal answer (about 1-3s) and well short of a cold start. */
export const SLOW_AFTER_MS = 6000;

export function useSlowRequest(active: boolean, afterMs: number = SLOW_AFTER_MS): boolean {
  const [slow, setSlow] = useState(false);

  // The reset belongs in the cleanup: it runs exactly when the wait ends -
  // the request settled, or the component went away - rather than on a later
  // render that has to notice it did.
  useEffect(() => {
    if (!active) return;
    const timer = setTimeout(() => setSlow(true), afterMs);
    return () => {
      clearTimeout(timer);
      setSlow(false);
    };
  }, [active, afterMs]);

  return slow;
}
