/**
 * Session state for one browser tab.
 *
 * The session id is a UUID generated when the tab loads and kept in React
 * state only - deliberately not in localStorage, so a refresh starts a clean
 * session with no active dataset.
 *
 * The answer cache lives here too, keyed by dataset + question. Navigating
 * between questions never calls the backend; only asking does, and only on a
 * miss. Replacing the dataset changes the key prefix, so old answers can
 * never be shown against new data.
 */

import { useCallback, useMemo, useRef, useState } from "react";
import * as api from "../api/client";
import { ApiError } from "../api/client";
import type { AskResponse, DatasetState } from "../types/api";

export type DatasetStatus = "idle" | "loading" | "ready" | "error";

function newSessionId(): string {
  return crypto.randomUUID();
}

export function useSession() {
  const [sessionId] = useState(newSessionId);
  const [dataset, setDataset] = useState<DatasetState | null>(null);
  const [datasetStatus, setDatasetStatus] = useState<DatasetStatus>("idle");
  const [datasetError, setDatasetError] = useState<string | null>(null);

  // A token that changes whenever the active dataset does, so cached answers
  // belong to exactly one dataset.
  const [datasetToken, setDatasetToken] = useState(0);
  const answers = useRef(new Map<string, AskResponse>());

  const [asking, setAsking] = useState(false);

  const replaceDataset = useCallback((next: DatasetState) => {
    answers.current.clear();
    setDataset(next);
    setDatasetToken((token) => token + 1);
    setDatasetStatus("ready");
    setDatasetError(null);
  }, []);

  const load = useCallback(
    async (loader: () => Promise<DatasetState>) => {
      setDatasetStatus("loading");
      setDatasetError(null);
      try {
        replaceDataset(await loader());
        return true;
      } catch (error) {
        // A failed load must not disturb the dataset already in use.
        setDatasetError(error instanceof ApiError ? error.message : "The dataset could not be loaded.");
        setDatasetStatus(dataset ? "ready" : "error");
        return false;
      }
    },
    [dataset, replaceDataset],
  );

  const loadAssessment = useCallback(
    () => load(() => api.loadAssessmentDataset(sessionId)),
    [load, sessionId],
  );

  const upload = useCallback(
    (file: File) => load(() => api.uploadDataset(sessionId, file)),
    [load, sessionId],
  );

  const cacheKey = useCallback((question: string) => `${datasetToken}::${question}`, [datasetToken]);

  const cachedAnswer = useCallback(
    (question: string) => answers.current.get(cacheKey(question)) ?? null,
    [cacheKey],
  );

  const ask = useCallback(
    async (question: string): Promise<AskResponse> => {
      const key = cacheKey(question);
      const cached = answers.current.get(key);
      if (cached) return cached;

      setAsking(true);
      try {
        const response = await api.askQuestion(sessionId, question);
        answers.current.set(key, response);
        return response;
      } catch (error) {
        // Transport failures are shown like any other non-answer, and are not
        // cached: the next attempt should really try again.
        return {
          status: "error",
          question,
          message: error instanceof ApiError ? error.message : "Something went wrong.",
        };
      } finally {
        setAsking(false);
      }
    },
    [cacheKey, sessionId],
  );

  return useMemo(
    () => ({
      sessionId,
      dataset,
      datasetStatus,
      datasetError,
      datasetToken,
      loadAssessment,
      upload,
      ask,
      cachedAnswer,
      asking,
    }),
    [sessionId, dataset, datasetStatus, datasetError, datasetToken, loadAssessment, upload, ask, cachedAnswer, asking],
  );
}

export type Session = ReturnType<typeof useSession>;
