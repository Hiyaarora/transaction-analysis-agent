/**
 * The only module that speaks HTTP.
 *
 * Components call typed functions; they never build a request. Every failure
 * arrives as an `ApiError` carrying a message fit to show a user - the server
 * sends {code, message}, and anything else (network down, HTML error page,
 * malformed JSON) is translated here rather than leaking a raw exception.
 */

import type { AskResponse, DatasetState, Health } from "../types/api";

export class ApiError extends Error {
  readonly code: string;
  readonly status: number;

  constructor(message: string, code = "request_failed", status = 0) {
    super(message);
    this.name = "ApiError";
    this.code = code;
    this.status = status;
  }
}

const TIMEOUT_MS = 90_000; // planning calls a model; give it room

async function request<T>(path: string, sessionId: string, init: RequestInit = {}): Promise<T> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), TIMEOUT_MS);

  let response: Response;
  try {
    response = await fetch(`/api${path}`, {
      ...init,
      signal: controller.signal,
      headers: { "X-Session-Id": sessionId, ...(init.headers ?? {}) },
    });
  } catch (cause) {
    const aborted = cause instanceof DOMException && cause.name === "AbortError";
    throw new ApiError(
      aborted
        ? "The request took too long and was cancelled."
        : "Could not reach the analysis server. Is it running?",
      aborted ? "timeout" : "offline",
    );
  } finally {
    clearTimeout(timer);
  }

  const body = await readJson(response);

  if (!response.ok) {
    const detail = (body as { detail?: unknown })?.detail;
    if (detail && typeof detail === "object" && "message" in detail) {
      const { code, message } = detail as { code?: string; message?: string };
      throw new ApiError(message ?? "The request failed.", code ?? "request_failed", response.status);
    }
    // FastAPI validation errors and anything unexpected.
    throw new ApiError(
      response.status === 422 ? "That request was not valid." : "The analysis server returned an error.",
      "request_failed",
      response.status,
    );
  }
  return body as T;
}

async function readJson(response: Response): Promise<unknown> {
  try {
    return await response.json();
  } catch {
    throw new ApiError("The analysis server sent a response we could not read.", "bad_response", response.status);
  }
}

export function getHealth(sessionId: string): Promise<Health> {
  return request<Health>("/health", sessionId);
}

export function loadAssessmentDataset(sessionId: string): Promise<DatasetState> {
  return request<DatasetState>("/dataset/assessment", sessionId, { method: "POST" });
}

export function uploadDataset(sessionId: string, file: File): Promise<DatasetState> {
  const form = new FormData();
  form.append("file", file);
  // No Content-Type header: the browser must set the multipart boundary.
  return request<DatasetState>("/dataset/upload", sessionId, { method: "POST", body: form });
}

export function getDataset(sessionId: string): Promise<DatasetState> {
  return request<DatasetState>("/dataset", sessionId);
}

/** Fetch the active dataset's CSV and hand it to the browser as a download.
 *
 *  A plain link cannot be used: the request needs the session header, so the
 *  bytes are fetched and then offered through a temporary object URL.
 */
export async function downloadDataset(sessionId: string, filename: string): Promise<void> {
  const response = await fetch("/api/dataset/download", { headers: { "X-Session-Id": sessionId } });
  if (!response.ok) throw new ApiError("The dataset could not be downloaded.", "download_failed", response.status);

  const url = URL.createObjectURL(await response.blob());
  try {
    const link = document.createElement("a");
    link.href = url;
    link.download = filename;
    link.click();
  } finally {
    URL.revokeObjectURL(url);
  }
}

export function askQuestion(sessionId: string, question: string): Promise<AskResponse> {
  return request<AskResponse>("/ask", sessionId, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });
}
