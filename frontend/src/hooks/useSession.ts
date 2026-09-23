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

/** Where the active dataset came from, so the panel can show the file in the
 *  place the button that loaded it used to be. */
export type DatasetSource = "assessment" | "upload" | null;

function newSessionId(): string {
  return crypto.randomUUID();
}

export function useSession() {
  const [sessionId] = useState(newSessionId);
  const [dataset, setDataset] = useState<DatasetState | null>(null);
  const [datasetStatus, setDatasetStatus] = useState<DatasetStatus>("idle");
  const [datasetError, setDatasetError] = useState<string | null>(null);
  const [datasetSource, setDatasetSource] = useState<DatasetSource>(null);

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
    async (loader: () => Promise<DatasetState>, source: Exclude<DatasetSource, null>) => {
      setDatasetStatus("loading");
      setDatasetError(null);
      try {
        replaceDataset(await loader());
        setDatasetSource(source);
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
    () => load(() => api.loadAssessmentDataset(sessionId), "assessment"),
    [load, sessionId],
  );

  const upload = useCallback(
    (file: File) => load(() => api.uploadDataset(sessionId, file), "upload"),
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
        // A failure is never remembered. An answer, a clarification and a
        // rejection are all determined by the question and the dataset, so
        // they will not change; an error is a moment in time - the provider
        // was busy - and asking again is the whole point of asking again.
        // Note this arrives as HTTP 200 with status "error", so it reaches
        // here rather than the catch below.
        if (response.status !== "error") answers.current.set(key, response);
        return response;
      } catch (error) {
        // A transport failure never got as far as a status, and is likewise
        // not cached.
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
      datasetSource,
      datasetToken,
      loadAssessment,
      upload,
      ask,
      cachedAnswer,
      asking,
    }),
    [sessionId, dataset, datasetStatus, datasetError, datasetSource, datasetToken, loadAssessment, upload,
     ask, cachedAnswer, asking],
  );
}

export type Session = ReturnType<typeof useSession>;
