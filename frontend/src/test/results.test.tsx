/**
 * The structured result display and the custom question box.
 *
 * Operation sentences are built from StepRecord fields, so the tests feed
 * invented steps and assert the wording - never a number the UI worked out
 * for itself.
 */

import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "../App";
import { describeStep } from "../components/operations";
import type { AskResponse, DatasetState, StepRecord } from "../types/api";

function step(tool: string, args: Record<string, unknown>, counts: Partial<StepRecord> = {}): StepRecord {
  return { tool, args, rows_in: null, rows_out: null, groups_in: null, groups_out: null, ...counts };
}

function datasetFixture(overrides: Partial<DatasetState> = {}): DatasetState {
  return {
    source_name: "fixture.csv",
    row_count: 12,
    columns: ["id", "date", "region", "product", "units", "unit_price", "discount"],
    categorical_values: { region: ["ZA", "ZB"], product: ["Widget"] },
    numeric_ranges: { units: [1, 9], unit_price: [10, 90], discount: [0, 0.5] },
    date_range: { start: "2031-02-03", end: "2031-08-09" },
    null_counts: {},
    parse_error_counts: {},
    extra_columns: [],
    supported_metrics: ["revenue"],
    questions: [],
    ...overrides,
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

async function loadThenAsk(answer: AskResponse, dataset = datasetFixture()) {
  fetchMock.mockResolvedValueOnce(jsonResponse(dataset));
  render(<App />);
  await userEvent.click(screen.getByRole("button", { name: /use assessment dataset/i }));
  await screen.findByRole("region", { name: /ask your own question/i });

  fetchMock.mockResolvedValueOnce(jsonResponse(answer));
  await userEvent.type(screen.getByRole("textbox", { name: /ask your own question/i }), "my question");
  await userEvent.click(screen.getByRole("button", { name: /^ask$/i }));
  return within(await screen.findByRole("region", { name: /result/i }));
}

// --- operation sentences ---------------------------------------------------------

describe("operation sentences", () => {
  it("describes a row filter with the rows it kept", () => {
    expect(
      describeStep(step("filter_rows", { column: "region", op: "eq", value: "ZA" }, { rows_in: 12, rows_out: 5 })),
    ).toBe("Filtered transactions where region = ZA (5 of 12 rows kept).");
  });

  it.each([
    [{ column: "units", op: "gt", value: 5 }, "units > 5"],
    [{ column: "units", op: "lte", value: 5 }, "units <= 5"],
    [{ column: "region", op: "neq", value: "ZA" }, "region != ZA"],
    [{ column: "product", op: "in", value: ["Widget", "Gadget"] }, "product is one of Widget, Gadget"],
    [{ column: "date", op: "between", value: ["2031-01-01", "2031-02-15"] }, "date between 2031-01-01 and 2031-02-15"],
    [{ column: "discount", op: "is_null" }, "discount is missing"],
    [{ column: "discount", op: "not_null" }, "discount is present"],
  ])("phrases %o", (args, expected) => {
    expect(describeStep(step("filter_rows", args, { rows_in: 1, rows_out: 1 }))).toContain(expected);
  });

  it("describes computing a derived metric with its formula", () => {
    expect(describeStep(step("compute_metric", { metric: "revenue" }, { rows_in: 5, rows_out: 5 }))).toBe(
      "Computed revenue for each of the 5 matching transactions (revenue = units * unit_price * (1 - discount)).",
    );
  });

  it("describes aggregations, including count", () => {
    expect(describeStep(step("aggregate", { column: "revenue", func: "sum" }, { rows_in: 5 }))).toBe(
      "Summed revenue over 5 matching transactions.",
    );
    expect(describeStep(step("aggregate", { column: "id", func: "count" }, { rows_in: 1 }))).toBe(
      "Counted the 1 matching transaction.",
    );
  });

  it("describes grouping and group filtering", () => {
    expect(
      describeStep(step("group_by", { by: "region", func: "sum", column: "revenue" }, { rows_in: 12, groups_out: 2 })),
    ).toBe("Grouped 12 transactions by region and summed revenue within each group (2 groups).");
    expect(describeStep(step("group_by", { by: "region", func: "count" }, { rows_in: 12, groups_out: 2 }))).toContain(
      "counted the transactions in each group",
    );
    expect(describeStep(step("filter_groups", { op: "gt", value: 500 }, { groups_in: 3, groups_out: 1 }))).toBe(
      "Kept groups whose value is greater than 500.00 (1 of 3 groups kept).",
    );
  });

  it("describes selecting an extreme", () => {
    expect(describeStep(step("select_extreme", { mode: "highest" }, { groups_in: 3, groups_out: 1 }))).toBe(
      "Selected the group with the highest value (3 groups compared).",
    );
  });
});

// --- the three sections ----------------------------------------------------------

const SCALAR_ANSWER: AskResponse = {
  status: "success",
  question: "my question",
  intent: "Total revenue for region ZA",
  execution: {
    kind: "scalar",
    steps: [
      step("filter_rows", { column: "region", op: "eq", value: "ZA" }, { rows_in: 12, rows_out: 5 }),
      step("compute_metric", { metric: "revenue" }, { rows_in: 5, rows_out: 5 }),
      step("aggregate", { column: "revenue", func: "sum" }, { rows_in: 5 }),
    ],
    value: 1234.5,
    rows_total: 5,
    rows_used: 5,
  },
};

describe("result display", () => {
  it("shows answer, operations and explanation", async () => {
    const result = await loadThenAsk(SCALAR_ANSWER);

    expect(result.getByText(/^answer$/i)).toBeInTheDocument();
    expect(result.getByText("Total revenue for region ZA")).toBeInTheDocument();
    expect(result.getByText("1,234.50")).toBeInTheDocument();

    expect(result.getByText(/operations performed/i)).toBeInTheDocument();
    const operations = result.getAllByRole("listitem");
    expect(operations).toHaveLength(3);
    expect(operations[0]).toHaveTextContent(/filtered transactions where region = ZA \(5 of 12 rows kept\)/i);

    expect(result.getByText(/^explanation$/i)).toBeInTheDocument();
    expect(result.getByText(/computed deterministically in python/i)).toBeInTheDocument();
    expect(result.getByText(/fixture\.csv/)).toBeInTheDocument();
  });

  it("stays quiet about usable rows when every row was usable", async () => {
    const result = await loadThenAsk(SCALAR_ANSWER);
    expect(result.queryByText(/were left out/i)).not.toBeInTheDocument();
    expect(result.queryByText(/computed from/i)).not.toBeInTheDocument();
  });

  it("says how many rows were usable when some were not", async () => {
    const result = await loadThenAsk({
      ...SCALAR_ANSWER,
      execution: { ...SCALAR_ANSWER.execution, kind: "scalar", rows_total: 5, rows_used: 3 },
    } as AskResponse);
    expect(result.getByText(/computed from 3 of the 5 matching transactions/i)).toBeInTheDocument();
    expect(result.getByText(/rather than treated as zero/i)).toBeInTheDocument();
  });

  it("uses singular wording when exactly one row was left out", async () => {
    const result = await loadThenAsk({
      ...SCALAR_ANSWER,
      execution: { ...SCALAR_ANSWER.execution, kind: "scalar", rows_total: 5, rows_used: 4 },
    } as AskResponse);
    expect(result.getByText(/1 had a missing value and was left out/i)).toBeInTheDocument();
  });

  it("shows a grouped result as a table with per-group counts", async () => {
    const result = await loadThenAsk({
      status: "success",
      question: "my question",
      intent: "Revenue by region",
      execution: {
        kind: "groups",
        steps: [step("group_by", { by: "region", func: "sum", column: "revenue" }, { rows_in: 12, groups_out: 2 })],
        by: "region",
        func: "sum",
        column: "revenue",
        rows: [
          { group: "ZA", value: 900, rows_total: 3, rows_used: 2 },
          { group: "ZB", value: null, rows_total: 1, rows_used: 0 },
        ],
        rows_without_group: 1,
        groups_without_value: 0,
      },
    });

    expect(result.getByText("ZA")).toBeInTheDocument();
    expect(result.getByText("900")).toBeInTheDocument();
    expect(result.getByText(/no value/i)).toBeInTheDocument();
    expect(result.getByText(/1 transaction had no group value/i)).toBeInTheDocument();
  });

  it("reports no data with the dataset's real date range", async () => {
    const result = await loadThenAsk({
      status: "no_data",
      question: "my question",
      intent: "Revenue in 1990",
      execution: {
        kind: "scalar",
        steps: [step("filter_rows", { column: "date", op: "between", value: ["1990-01-01", "1990-12-31"] }, { rows_in: 12, rows_out: 0 })],
        value: null,
        rows_total: 0,
        rows_used: 0,
      },
    });

    expect(result.getByText(/no data/i)).toBeInTheDocument();
    expect(result.getByText(/2031-02-03/)).toBeInTheDocument();
    expect(result.getByText(/2031-08-09/)).toBeInTheDocument();
    expect(result.queryByText(/^0$/)).not.toBeInTheDocument(); // never reported as zero
  });
});

// --- custom questions -------------------------------------------------------------

describe("custom questions", () => {
  it("sends exactly what was typed", async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse(datasetFixture()));
    render(<App />);
    await userEvent.click(screen.getByRole("button", { name: /use assessment dataset/i }));
    await screen.findByRole("region", { name: /ask your own question/i });

    fetchMock.mockResolvedValueOnce(jsonResponse(SCALAR_ANSWER));
    await userEvent.type(
      screen.getByRole("textbox", { name: /ask your own question/i }),
      "Which region invented the most widgets?",
    );
    await userEvent.click(screen.getByRole("button", { name: /^ask$/i }));

    await waitFor(() => expect(fetchMock.mock.calls.some(([url]) => url === "/api/ask")).toBe(true));
    const askCall = fetchMock.mock.calls.find(([url]) => url === "/api/ask")!;
    expect(JSON.parse(askCall[1].body)).toEqual({ question: "Which region invented the most widgets?" });
  });

  it("will not send an empty question", async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse(datasetFixture()));
    render(<App />);
    await userEvent.click(screen.getByRole("button", { name: /use assessment dataset/i }));
    await screen.findByRole("region", { name: /ask your own question/i });

    expect(screen.getByRole("button", { name: /^ask$/i })).toBeDisabled();
    await userEvent.type(screen.getByRole("textbox", { name: /ask your own question/i }), "   ");
    expect(screen.getByRole("button", { name: /^ask$/i })).toBeDisabled();
  });

  it("shows a rejection from a custom question as itself", async () => {
    const result = await loadThenAsk({
      status: "rejected",
      question: "my question",
      message: "I cannot run code or access files.",
    });
    expect(result.getByText(/request rejected/i)).toBeInTheDocument();
    expect(result.getByText(/cannot run code/i)).toBeInTheDocument();
    expect(result.queryByText(/operations performed/i)).not.toBeInTheDocument();
  });
});

describe("when the backend is unreachable", () => {
  it("reports an ask that could not be sent, without inventing an answer", async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse(datasetFixture()));
    render(<App />);
    await userEvent.click(screen.getByRole("button", { name: /use assessment dataset/i }));
    await screen.findByRole("region", { name: /ask your own question/i });

    fetchMock.mockRejectedValueOnce(new TypeError("Failed to fetch"));
    await userEvent.type(screen.getByRole("textbox", { name: /ask your own question/i }), "anything");
    await userEvent.click(screen.getByRole("button", { name: /^ask$/i }));

    const result = within(await screen.findByRole("region", { name: /result/i }));
    expect(result.getByText(/something went wrong/i)).toBeInTheDocument();
    expect(result.getByText(/could not reach the analysis server/i)).toBeInTheDocument();
    expect(result.queryByText(/operations performed/i)).not.toBeInTheDocument();
  });

  it("does not cache a failure, so the next attempt really retries", async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse(datasetFixture()));
    render(<App />);
    await userEvent.click(screen.getByRole("button", { name: /use assessment dataset/i }));
    await screen.findByRole("region", { name: /ask your own question/i });

    const input = screen.getByRole("textbox", { name: /ask your own question/i });
    await userEvent.type(input, "anything");

    fetchMock.mockRejectedValueOnce(new TypeError("Failed to fetch"));
    await userEvent.click(screen.getByRole("button", { name: /^ask$/i }));
    await screen.findByText(/something went wrong/i);

    fetchMock.mockResolvedValueOnce(jsonResponse(SCALAR_ANSWER));
    await userEvent.click(screen.getByRole("button", { name: /^ask$/i }));

    expect(await screen.findByText("1,234.50")).toBeInTheDocument();
    expect(fetchMock.mock.calls.filter(([url]) => url === "/api/ask")).toHaveLength(2);
  });
});

describe("replacing the dataset", () => {
  it("clears a custom answer computed from the previous dataset", async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse(datasetFixture()));
    render(<App />);
    await userEvent.click(screen.getByRole("button", { name: /use assessment dataset/i }));
    await screen.findByRole("region", { name: /ask your own question/i });

    fetchMock.mockResolvedValueOnce(jsonResponse(SCALAR_ANSWER));
    await userEvent.type(screen.getByRole("textbox", { name: /ask your own question/i }), "my question");
    await userEvent.click(screen.getByRole("button", { name: /^ask$/i }));
    expect(await screen.findByText("1,234.50")).toBeInTheDocument();

    // A different file is loaded: an answer computed from the old one must not
    // remain on screen next to the new dataset's profile.
    fetchMock.mockResolvedValueOnce(
      jsonResponse(datasetFixture({ source_name: "different.csv", row_count: 3 })),
    );
    await userEvent.upload(
      screen.getByLabelText(/upload a csv file/i),
      new File(["id\n"], "different.csv", { type: "text/csv" }),
    );

    await waitFor(() => expect(screen.queryByRole("region", { name: /result/i })).not.toBeInTheDocument());
    expect(screen.queryByText("1,234.50")).not.toBeInTheDocument();
  });
});

describe("while a new question is processing", () => {
  it("clears the previous answer instead of leaving it on screen", async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse(datasetFixture()));
    render(<App />);
    await userEvent.click(screen.getByRole("button", { name: /use assessment dataset/i }));
    await screen.findByRole("region", { name: /ask your own question/i });

    const input = screen.getByRole("textbox", { name: /ask your own question/i });
    fetchMock.mockResolvedValueOnce(jsonResponse(SCALAR_ANSWER));
    await userEvent.type(input, "first question");
    await userEvent.click(screen.getByRole("button", { name: /^ask$/i }));
    expect(await screen.findByText("1,234.50")).toBeInTheDocument();

    // A second, different question: hold the response open and check what is
    // on screen mid-flight.
    let release: (value: Response) => void = () => {};
    fetchMock.mockReturnValueOnce(new Promise<Response>((resolve) => (release = resolve)));
    await userEvent.clear(input);
    await userEvent.type(input, "second question");
    await userEvent.click(screen.getByRole("button", { name: /^ask$/i }));

    expect(screen.queryByText("1,234.50")).not.toBeInTheDocument();
    expect(screen.queryByRole("region", { name: /result/i })).not.toBeInTheDocument();
    expect(screen.getByText(/planning/i)).toBeInTheDocument();

    release(jsonResponse({ ...SCALAR_ANSWER, intent: "Second answer" }));
    expect(await screen.findByText("Second answer")).toBeInTheDocument();
  });

  it("clears a previous failure when the question is retried", async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse(datasetFixture()));
    render(<App />);
    await userEvent.click(screen.getByRole("button", { name: /use assessment dataset/i }));
    await screen.findByRole("region", { name: /ask your own question/i });

    const input = screen.getByRole("textbox", { name: /ask your own question/i });
    await userEvent.type(input, "a question");
    fetchMock.mockRejectedValueOnce(new TypeError("Failed to fetch"));
    await userEvent.click(screen.getByRole("button", { name: /^ask$/i }));
    expect(await screen.findByText(/something went wrong/i)).toBeInTheDocument();

    let release: (value: Response) => void = () => {};
    fetchMock.mockReturnValueOnce(new Promise<Response>((resolve) => (release = resolve)));
    await userEvent.click(screen.getByRole("button", { name: /^ask$/i }));

    expect(screen.queryByText(/something went wrong/i)).not.toBeInTheDocument();

    release(jsonResponse(SCALAR_ANSWER));
    expect(await screen.findByText("1,234.50")).toBeInTheDocument();
  });
});
