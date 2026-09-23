/**
 * The long-wait notice.
 *
 * The deployed service sleeps after a quiet period, so the request that wakes
 * it can take tens of seconds. These tests assert the UI explains that wait
 * instead of showing a spinner that is indistinguishable from a hang - and,
 * just as importantly, that it says nothing during a normal fast answer.
 */

import { act, fireEvent, render, renderHook, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "../App";
import { SLOW_AFTER_MS, useSlowRequest } from "../hooks/useSlowRequest";

beforeEach(() => {
  vi.useFakeTimers();
});

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe("useSlowRequest", () => {
  it("stays false while a request is still within the normal window", () => {
    const { result } = renderHook(() => useSlowRequest(true));

    act(() => void vi.advanceTimersByTime(SLOW_AFTER_MS - 1));
    expect(result.current).toBe(false);
  });

  it("becomes true once the wait passes the threshold", () => {
    const { result } = renderHook(() => useSlowRequest(true));

    act(() => void vi.advanceTimersByTime(SLOW_AFTER_MS));
    expect(result.current).toBe(true);
  });

  it("resets when the request finishes, so the next one starts quiet", () => {
    const { result, rerender } = renderHook(({ active }) => useSlowRequest(active), {
      initialProps: { active: true },
    });
    act(() => void vi.advanceTimersByTime(SLOW_AFTER_MS));
    expect(result.current).toBe(true);

    rerender({ active: false });
    expect(result.current).toBe(false);
  });

  it("never fires for a request that was over before the threshold", () => {
    const { result, rerender } = renderHook(({ active }) => useSlowRequest(active), {
      initialProps: { active: true },
    });
    act(() => void vi.advanceTimersByTime(1000));
    rerender({ active: false });

    act(() => void vi.advanceTimersByTime(SLOW_AFTER_MS * 2));
    expect(result.current).toBe(false);
  });
});

describe("in the interface", () => {
  it("explains the wait while a dataset is loading from a sleeping server", () => {
    // A request that never settles: the server is still starting.
    vi.stubGlobal("fetch", vi.fn(() => new Promise<Response>(() => {})));
    vi.stubGlobal("crypto", { ...crypto, randomUUID: () => "session-under-test-0001" });

    render(<App />);
    fireEvent.click(screen.getByRole("button", { name: /use assessment dataset/i }));

    expect(screen.queryByText(/up to a minute/i)).not.toBeInTheDocument();

    act(() => void vi.advanceTimersByTime(SLOW_AFTER_MS));
    expect(screen.getByText(/up to a minute/i)).toBeInTheDocument();
  });
});
