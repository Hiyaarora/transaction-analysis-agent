/**
 * The assessment question flow.
 *
 * The fixtures invent their own questions, and never ten of them, because the
 * UI must read the list and its length from the dataset rather than assume
 * the supplied file's shape.
 */

import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "../App";
import type { AskResponse, DatasetState } from "../types/api";

const QUESTIONS = ["First invented question?", "Second invented question?", "Third invented question?"];

function datasetFixture(overrides: Partial<DatasetState> = {}): DatasetState {
  return {
    source_name: "fixture.csv",
    row_count: 5,
    columns: ["id", "date", "region", "product", "units", "unit_price", "discount"],
    categorical_values: { region: ["ZA"], product: ["Widget"] },
    numeric_ranges: { units: [1, 9], unit_price: [10, 90], discount: [0, 0.5] },
    date_range: { start: "2031-02-03", end: "2031-08-09" },
    null_counts: {},
    parse_error_counts: {},
    extra_columns: [],
    supported_metrics: ["revenue"],
    questions: QUESTIONS,
    ...overrides,
  };
}

function answerFixture(question: string, value: number): AskResponse {
  return {
    status: "success",
    question,
    intent: `Intent for ${question}`,
    execution: {
      kind: "scalar",
      steps: [{ tool: "aggregate", args: { column: "units", func: "sum" }, rows_in: 5, rows_out: null, groups_in: null, groups_out: null }],
      value,
      rows_total: 5,
      rows_used: 5,
    },
  };
}

function jsonResponse(body: unknown, status = 200): Response {
  return { ok: status < 400, status, json: async () => body } as Response;
}

let fetchMock: ReturnType<typeof vi.fn>;

beforeEach(() => {
  fetchMock = vi.fn();
  vi.stubGlobal("fetch", fetchMock);
  vi.stubGlobal("crypto", { ...crypto, randomUUID: () => "session-under-test-0001" });
});

afterEach(() => vi.unstubAllGlobals());

/** Load a dataset, then return the assessment section. */
async function loadDataset(dataset: DatasetState = datasetFixture()) {
  fetchMock.mockResolvedValueOnce(jsonResponse(dataset));
  render(<App />);
  await userEvent.click(screen.getByRole("button", { name: /use assessment dataset/i }));
  return within(await screen.findByRole("region", { name: /assessment questions/i }));
}

const askCalls = () => fetchMock.mock.calls.filter(([url]) => url === "/api/ask");

describe("question navigation", () => {
  it("shows the first question and the total, both from the dataset", async () => {
    const panel = await loadDataset();
    expect(panel.getByText(/question 1 of 3/i)).toBeInTheDocument();
    expect(panel.getByText(QUESTIONS[0])).toBeInTheDocument();
  });

  it("adapts to a dataset with a different number of questions", async () => {
    const panel = await loadDataset(datasetFixture({ questions: ["Only one question?"] }));
    expect(panel.getByText(/question 1 of 1/i)).toBeInTheDocument();
    expect(panel.getByText("Only one question?")).toBeInTheDocument();
  });

  it("hides the assessment section when the dataset carries no questions", async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse(datasetFixture({ questions: [] })));
    render(<App />);
    await userEvent.click(screen.getByRole("button", { name: /use assessment dataset/i }));
    await screen.findByRole("region", { name: /ask your own question/i });

    expect(screen.queryByRole("region", { name: /assessment questions/i })).not.toBeInTheDocument();
  });

  it("moves forwards and backwards through the list", async () => {
    const panel = await loadDataset();

    await userEvent.click(panel.getByRole("button", { name: /next/i }));
    expect(panel.getByText(/question 2 of 3/i)).toBeInTheDocument();
    expect(panel.getByText(QUESTIONS[1])).toBeInTheDocument();

    await userEvent.click(panel.getByRole("button", { name: /previous/i }));
    expect(panel.getByText(/question 1 of 3/i)).toBeInTheDocument();
    expect(panel.getByText(QUESTIONS[0])).toBeInTheDocument();
  });

  it("stops at both ends", async () => {
    const panel = await loadDataset();
    expect(panel.getByRole("button", { name: /previous/i })).toBeDisabled();

    await userEvent.click(panel.getByRole("button", { name: /next/i }));
    await userEvent.click(panel.getByRole("button", { name: /next/i }));
    expect(panel.getByText(/question 3 of 3/i)).toBeInTheDocument();
    expect(panel.getByRole("button", { name: /next/i })).toBeDisabled();
  });

  it("never calls the API while navigating", async () => {
    const panel = await loadDataset();
    await userEvent.click(panel.getByRole("button", { name: /next/i }));
    await userEvent.click(panel.getByRole("button", { name: /next/i }));
    await userEvent.click(panel.getByRole("button", { name: /previous/i }));

    expect(askCalls()).toHaveLength(0);
  });
});

describe("asking an assessment question", () => {
  it("sends the question text from the dataset and shows the answer", async () => {
    const panel = await loadDataset();
    fetchMock.mockResolvedValueOnce(jsonResponse(answerFixture(QUESTIONS[0], 42)));

    await userEvent.click(panel.getByRole("button", { name: /^ask question$/i }));

    expect(await panel.findByText("42")).toBeInTheDocument();
    expect(askCalls()).toHaveLength(1);
    expect(JSON.parse(askCalls()[0][1].body)).toEqual({ question: QUESTIONS[0] });
  });

  it("shows progress while the request is in flight and disables the button", async () => {
    const panel = await loadDataset();
    let release: (value: Response) => void = () => {};
    fetchMock.mockReturnValueOnce(new Promise<Response>((resolve) => (release = resolve)));

    await userEvent.click(panel.getByRole("button", { name: /^ask question$/i }));

    expect(panel.getByText(/planning/i)).toBeInTheDocument();
    expect(panel.getByRole("button", { name: /analysing|analyzing|planning/i })).toBeDisabled();

    release(jsonResponse(answerFixture(QUESTIONS[0], 7)));
    expect(await panel.findByText("7")).toBeInTheDocument();
  });

  it("does not call the API again for a question already answered", async () => {
    const panel = await loadDataset();
    fetchMock.mockResolvedValueOnce(jsonResponse(answerFixture(QUESTIONS[0], 42)));
    await userEvent.click(panel.getByRole("button", { name: /^ask question$/i }));
    await panel.findByText("42");

    // Leave the question and come back: the stored answer reappears with no request.
    await userEvent.click(panel.getByRole("button", { name: /next/i }));
    expect(panel.queryByText("42")).not.toBeInTheDocument();
    await userEvent.click(panel.getByRole("button", { name: /previous/i }));

    expect(await panel.findByText("42")).toBeInTheDocument();
    expect(askCalls()).toHaveLength(1);

    // Asking it again still costs nothing.
    await userEvent.click(panel.getByRole("button", { name: /^ask question$/i }));
    await waitFor(() => expect(askCalls()).toHaveLength(1));
  });

  it("asks again after the dataset is replaced", async () => {
    const panel = await loadDataset();
    fetchMock.mockResolvedValueOnce(jsonResponse(answerFixture(QUESTIONS[0], 42)));
    await userEvent.click(panel.getByRole("button", { name: /^ask question$/i }));
    await panel.findByText("42");

    // A new dataset with the same question text: the old answer must not be reused.
    fetchMock.mockResolvedValueOnce(jsonResponse(datasetFixture({ source_name: "other.csv" })));
    await userEvent.upload(
      screen.getByLabelText(/upload a csv file/i),
      new File(["id\n"], "other.csv", { type: "text/csv" }),
    );
    const reloaded = within(await screen.findByRole("region", { name: /assessment questions/i }));
    expect(reloaded.queryByText("42")).not.toBeInTheDocument();

    fetchMock.mockResolvedValueOnce(jsonResponse(answerFixture(QUESTIONS[0], 99)));
    await userEvent.click(reloaded.getByRole("button", { name: /^ask question$/i }));

    expect(await reloaded.findByText("99")).toBeInTheDocument();
    expect(askCalls()).toHaveLength(2);
  });

  it("shows a clarification instead of an answer", async () => {
    const panel = await loadDataset();
    fetchMock.mockResolvedValueOnce(
      jsonResponse({ status: "clarification_required", question: QUESTIONS[0], message: "Did you mean March?" }),
    );
    await userEvent.click(panel.getByRole("button", { name: /^ask question$/i }));

    expect(await panel.findByText(/clarification required/i)).toBeInTheDocument();
    expect(panel.getByText("Did you mean March?")).toBeInTheDocument();
  });

  it("shows a rejection instead of an answer", async () => {
    const panel = await loadDataset();
    fetchMock.mockResolvedValueOnce(
      jsonResponse({ status: "rejected", question: QUESTIONS[0], message: "I cannot run code." }),
    );
    await userEvent.click(panel.getByRole("button", { name: /^ask question$/i }));

    expect(await panel.findByText(/request rejected/i)).toBeInTheDocument();
    expect(panel.getByText("I cannot run code.")).toBeInTheDocument();
  });

  it("reports a provider failure without pretending it answered", async () => {
    const panel = await loadDataset();
    fetchMock.mockResolvedValueOnce(
      jsonResponse({ status: "error", question: QUESTIONS[0], message: "Gemini error 429: quota exceeded" }),
    );
    await userEvent.click(panel.getByRole("button", { name: /^ask question$/i }));

    expect(await panel.findByText(/something went wrong/i)).toBeInTheDocument();
    expect(panel.getByText(/quota exceeded/i)).toBeInTheDocument();
  });
});
