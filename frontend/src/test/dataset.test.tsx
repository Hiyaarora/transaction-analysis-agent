/**
 * Dataset selection and upload, driven through the real components.
 *
 * `fetch` is stubbed, so these assert what the UI does with an API response -
 * never what the analysis engine computes. The fixtures invent their own
 * questions, which is also how we prove the UI reads them from the response.
 *
 * The profile is no longer displayed, so "a dataset is loaded" is observed
 * through behaviour: the question sections appear, carrying that file's
 * questions.
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
    null_counts: { discount: 2 },
    parse_error_counts: {},
    extra_columns: [],
    supported_metrics: ["units", "unit_price", "discount", "revenue"],
    questions: ["Invented question one?", "Invented question two?"],
    ...overrides,
  };
}

function jsonResponse(body: unknown, status = 200): Response {
  return { ok: status < 400, status, json: async () => body } as Response;
}

/** The assessment section, which only exists once a dataset is active. */
async function questionPanel() {
  return within(await screen.findByRole("region", { name: /assessment questions/i }));
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

  it("loads the assessment dataset and shows the questions it carries", async () => {
    fetchMock.mockResolvedValue(jsonResponse(datasetFixture()));
    render(<App />);

    await userEvent.click(screen.getByRole("button", { name: /use assessment dataset/i }));

    const panel = await questionPanel();
    expect(panel.getByText("Invented question one?")).toBeInTheDocument();
    expect(panel.getByText(/question 1 of 2/i)).toBeInTheDocument();
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

  it("shows a clean message when the server is unreachable", async () => {
    fetchMock.mockRejectedValue(new TypeError("Failed to fetch"));
    render(<App />);
    await userEvent.click(screen.getByRole("button", { name: /use assessment dataset/i }));

    expect(await screen.findByRole("alert")).toHaveTextContent(/could not reach the analysis server/i);
    expect(screen.getByText(/no dataset loaded/i)).toBeInTheDocument();
  });
});

describe("uploading a dataset", () => {
  const csv = () => new File(["id,date\n"], "my-upload.csv", { type: "text/csv" });

  it("replaces the active dataset with the uploaded one", async () => {
    fetchMock
      .mockResolvedValueOnce(jsonResponse(datasetFixture()))
      .mockResolvedValueOnce(
        jsonResponse(
          datasetFixture({
            source_name: "my-upload.csv",
            questions: ["A question only the uploaded file asks?"],
          }),
        ),
      );
    render(<App />);

    await userEvent.click(screen.getByRole("button", { name: /use assessment dataset/i }));
    expect((await questionPanel()).getByText("Invented question one?")).toBeInTheDocument();

    await userEvent.upload(screen.getByLabelText(/upload a csv file/i), csv());

    const panel = await questionPanel();
    await waitFor(() =>
      expect(panel.getByText("A question only the uploaded file asks?")).toBeInTheDocument(),
    );
    expect(panel.getByText(/question 1 of 1/i)).toBeInTheDocument();
    expect(panel.queryByText("Invented question one?")).not.toBeInTheDocument();
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
      .mockResolvedValueOnce(jsonResponse(datasetFixture()))
      .mockResolvedValueOnce(
        jsonResponse({ detail: { code: "invalid_dataset", message: "Missing required column(s): product" } }, 400),
      );
    render(<App />);

    await userEvent.click(screen.getByRole("button", { name: /use assessment dataset/i }));
    await questionPanel();

    await userEvent.upload(screen.getByLabelText(/upload a csv file/i), csv());

    expect(await screen.findByRole("alert")).toHaveTextContent(/missing required column/i);
    // The first dataset's questions are still the ones on offer.
    expect((await questionPanel()).getByText("Invented question one?")).toBeInTheDocument();
  });
});

describe("the file in use", () => {
  it("names the loaded file on the dataset tile", async () => {
    fetchMock.mockResolvedValue(jsonResponse(datasetFixture({ source_name: "my-upload.csv" })));
    render(<App />);
    await userEvent.upload(
      screen.getByLabelText(/upload a csv file/i),
      new File(["id\n"], "my-upload.csv", { type: "text/csv" }),
    );

    expect(await screen.findByText("my-upload.csv")).toBeInTheDocument();
    // On the existing tile, not a second one.
    expect(screen.queryByRole("region", { name: /loaded dataset/i })).not.toBeInTheDocument();
  });

  it("offers no download until something is loaded", () => {
    render(<App />);
    expect(screen.queryByRole("button", { name: /download/i })).not.toBeInTheDocument();
  });

  it("downloads the csv under its own name", async () => {
    const click = vi.fn();
    const created: string[] = [];
    vi.spyOn(document, "createElement").mockImplementation(((tag: string) => {
      if (tag !== "a") return document.createElementNS("http://www.w3.org/1999/xhtml", tag);
      const anchor = { href: "", download: "", click } as unknown as HTMLAnchorElement;
      return anchor;
    }) as typeof document.createElement);
    vi.stubGlobal("URL", {
      ...URL,
      createObjectURL: (blob: Blob) => {
        created.push("blob-url");
        void blob;
        return "blob:fake";
      },
      revokeObjectURL: () => {},
    });

    fetchMock.mockResolvedValueOnce(jsonResponse(datasetFixture({ source_name: "mine.csv" })));
    render(<App />);
    await userEvent.click(screen.getByRole("button", { name: /use assessment dataset/i }));
    await screen.findByText("mine.csv");

    fetchMock.mockResolvedValueOnce({
      ok: true,
      status: 200,
      blob: async () => new Blob(["id,date\n"], { type: "text/csv" }),
    } as Response);
    await userEvent.click(screen.getByRole("button", { name: /download mine\.csv/i }));

    await waitFor(() => expect(click).toHaveBeenCalled());
    const [url, init] = fetchMock.mock.calls.find(([u]) => u === "/api/dataset/download")!;
    expect(url).toBe("/api/dataset/download");
    expect(init.headers["X-Session-Id"]).toBe("session-under-test-0001");
    expect(created).toHaveLength(1);
    vi.restoreAllMocks();
  });

  it("reports a download that failed, without breaking the page", async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse(datasetFixture()));
    render(<App />);
    await userEvent.click(screen.getByRole("button", { name: /use assessment dataset/i }));
    await screen.findByText("fixture.csv");

    fetchMock.mockResolvedValueOnce({ ok: false, status: 409, blob: async () => new Blob() } as Response);
    await userEvent.click(screen.getByRole("button", { name: /download/i }));

    expect(await screen.findByRole("alert")).toHaveTextContent(/could not be downloaded/i);
    expect(screen.getByRole("button", { name: /download/i })).toBeInTheDocument();
  });
});
