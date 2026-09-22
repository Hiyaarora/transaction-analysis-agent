/**
 * Dataset selection and upload, driven through the real components.
 *
 * `fetch` is stubbed, so these assert what the UI does with an API response -
 * never what the analysis engine computes. Nothing about regions, products or
 * counts is assumed: the fixtures invent their own values, which is also how
 * we prove the UI reads them from the response.
 */

import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "../App";
import type { DatasetState } from "../types/api";

function datasetFixture(overrides: Partial<DatasetState> = {}): DatasetState {
  return {
    source_name: "fixture.csv",
    row_count: 7,
    columns: ["id", "date", "region", "product", "units", "unit_price", "discount"],
    categorical_values: { region: ["ZA", "ZB"], product: ["Widget", "Gadget", "Doohickey"] },
    numeric_ranges: { units: [1, 9], unit_price: [10, 90], discount: [0, 0.5] },
    date_range: { start: "2031-02-03", end: "2031-08-09" },
    null_counts: { id: 0, date: 0, region: 0, product: 0, units: 0, unit_price: 0, discount: 2 },
    parse_error_counts: {},
    extra_columns: [],
    supported_metrics: ["units", "unit_price", "discount", "revenue"],
    questions: ["Invented question one?", "Invented question two?"],
    ...overrides,
  };
}

/** The panel showing the dataset actually in use (the card above it repeats
 *  the assessment file name, which would otherwise be ambiguous). */
async function activePanel() {
  return within(await screen.findByRole("region", { name: /active dataset/i }));
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

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("loading a dataset", () => {
  it("starts with nothing loaded", () => {
    render(<App />);
    expect(screen.getByText(/no dataset loaded/i)).toBeInTheDocument();
  });

  it("loads the assessment dataset and shows what came back", async () => {
    fetchMock.mockResolvedValue(jsonResponse(datasetFixture({ source_name: "project_4.csv" })));
    render(<App />);

    await userEvent.click(screen.getByRole("button", { name: /use assessment dataset/i }));

    const active = await activePanel();
    expect(active.getByText("project_4.csv")).toBeInTheDocument();
    expect(active.getByText("7")).toBeInTheDocument();
    expect(active.getByText("ZA")).toBeInTheDocument();
    expect(active.getByText("Doohickey")).toBeInTheDocument();
    expect(active.getByText(/2031-02-03/)).toBeInTheDocument();
    expect(screen.queryByText(/no dataset loaded/i)).not.toBeInTheDocument();
  });

  it("sends the session id with the request", async () => {
    fetchMock.mockResolvedValue(jsonResponse(datasetFixture()));
    render(<App />);
    await userEvent.click(screen.getByRole("button", { name: /use assessment dataset/i }));

    await waitFor(() => expect(fetchMock).toHaveBeenCalled());
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe("/api/dataset/assessment");
    expect(init.method).toBe("POST");
    expect(init.headers["X-Session-Id"]).toBe("session-under-test-0001");
  });

  it("reports missing values from the profile", async () => {
    fetchMock.mockResolvedValue(jsonResponse(datasetFixture()));
    render(<App />);
    await userEvent.click(screen.getByRole("button", { name: /use assessment dataset/i }));

    expect(await screen.findByText(/discount \(2\)/)).toBeInTheDocument();
  });

  it("names columns that exist but cannot be analysed", async () => {
    fetchMock.mockResolvedValue(jsonResponse(datasetFixture({ extra_columns: ["zone"] })));
    render(<App />);
    await userEvent.click(screen.getByRole("button", { name: /use assessment dataset/i }));

    expect(await screen.findByText(/not analysable/i)).toBeInTheDocument();
    expect(screen.getByText("zone")).toBeInTheDocument();
  });

  it("shows a clean message when the server is unreachable", async () => {
    fetchMock.mockRejectedValue(new TypeError("Failed to fetch"));
    render(<App />);
    await userEvent.click(screen.getByRole("button", { name: /use assessment dataset/i }));

    expect(await screen.findByRole("alert")).toHaveTextContent(/could not reach the analysis server/i);
  });
});

describe("uploading a dataset", () => {
  const csv = () => new File(["id,date\n"], "my-upload.csv", { type: "text/csv" });

  it("replaces the active dataset with the uploaded one", async () => {
    fetchMock
      .mockResolvedValueOnce(jsonResponse(datasetFixture({ source_name: "project_4.csv" })))
      .mockResolvedValueOnce(
        jsonResponse(
          datasetFixture({
            source_name: "my-upload.csv",
            row_count: 3,
            categorical_values: { region: ["QQ"], product: ["Sprocket"] },
          }),
        ),
      );
    render(<App />);

    await userEvent.click(screen.getByRole("button", { name: /use assessment dataset/i }));
    expect((await activePanel()).getByText("project_4.csv")).toBeInTheDocument();

    await userEvent.upload(screen.getByLabelText(/upload a csv file/i), csv());

    await waitFor(async () => expect((await activePanel()).getByText("my-upload.csv")).toBeInTheDocument());
    const active = await activePanel();
    expect(active.getByText("QQ")).toBeInTheDocument();
    expect(active.queryByText("project_4.csv")).not.toBeInTheDocument();
    expect(active.queryByText("ZA")).not.toBeInTheDocument();
  });

  it("posts the file as multipart form data", async () => {
    fetchMock.mockResolvedValue(jsonResponse(datasetFixture()));
    render(<App />);
    await userEvent.upload(screen.getByLabelText(/upload a csv file/i), csv());

    await waitFor(() => expect(fetchMock).toHaveBeenCalled());
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe("/api/dataset/upload");
    expect(init.body).toBeInstanceOf(FormData);
    // The browser must set the multipart boundary itself.
    expect(init.headers["Content-Type"]).toBeUndefined();
  });

  it("keeps the previous dataset when an upload is rejected", async () => {
    fetchMock
      .mockResolvedValueOnce(jsonResponse(datasetFixture({ source_name: "project_4.csv" })))
      .mockResolvedValueOnce(
        jsonResponse({ detail: { code: "invalid_dataset", message: "Missing required column(s): product" } }, 400),
      );
    render(<App />);

    await userEvent.click(screen.getByRole("button", { name: /use assessment dataset/i }));
    await activePanel();

    await userEvent.upload(screen.getByLabelText(/upload a csv file/i), csv());

    expect(await screen.findByRole("alert")).toHaveTextContent(/missing required column/i);
    expect((await activePanel()).getByText("project_4.csv")).toBeInTheDocument(); // still active
  });
});
