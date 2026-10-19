import { describe, expect, it } from "vitest";
import {
  apiErrorFromPayload,
  assistantMessageFromPayload,
  errorMessageFromUnknown,
  healthStatusFromPayload,
  isHealthCheckCurrent,
  isRequestCurrent,
} from "./chatApi";

describe("apiErrorFromPayload", () => {
  it("uses a non-empty backend detail", () => {
    expect(apiErrorFromPayload(429, { detail: "Rate limit reached" })).toBe("Rate limit reached");
  });

  it.each([null, [], {}, { detail: "" }, { detail: "  " }, { detail: 42 }])(
    "falls back to the HTTP status for unusable payloads: %j",
    (payload) => expect(apiErrorFromPayload(503, payload)).toBe("TARA backend returned HTTP 503."),
  );

  it("formats FastAPI validation-error arrays into readable messages", () => {
    expect(apiErrorFromPayload(422, {
      detail: [
        { loc: ["body", "message"], msg: "Field required", type: "missing" },
        { loc: ["body", "other"], msg: "Must be text", type: "string_type" },
      ],
    })).toBe("Field required Must be text");
  });

  it("limits validation-error output to three messages", () => {
    expect(apiErrorFromPayload(422, {
      detail: [
        { msg: "First" }, { msg: "Second" }, { msg: "Third" }, { msg: "Fourth" },
      ],
    })).toBe("First Second Third");
  });

  it("ignores malformed validation-error entries", () => {
    expect(apiErrorFromPayload(422, { detail: [null, { msg: 42 }, {}] }))
      .toBe("TARA backend returned HTTP 422.");
  });

  it("trims backend detail", () => {
    expect(apiErrorFromPayload(400, { detail: "  Bad request  " })).toBe("Bad request");
  });
});

describe("assistantMessageFromPayload", () => {
  it("returns trimmed assistant text", () => {
    expect(assistantMessageFromPayload({ message: "  Hello TARA  " })).toBe("Hello TARA");
  });

  it.each([null, [], {}, { message: 12 }, { message: null }, { message: "" }, { message: "   " }])(
    "rejects malformed or empty assistant responses: %j",
    (payload) => expect(() => assistantMessageFromPayload(payload)).toThrow(),
  );
});

describe("errorMessageFromUnknown", () => {
  it("returns a trimmed Error message", () => {
    expect(errorMessageFromUnknown(new Error("  Network unavailable  "))).toBe("Network unavailable");
  });

  it("returns a useful non-empty string when a string is thrown", () => {
    expect(errorMessageFromUnknown("  Request failed  ")).toBe("Request failed");
  });

  it.each([null, undefined, 0, false, "", "   ", {}])("uses a fallback for unusable thrown values: %j", (error) => {
    expect(errorMessageFromUnknown(error)).toBe("An unexpected error occurred.");
  });

  it("allows a caller-provided fallback", () => {
    expect(errorMessageFromUnknown(null, "Please try again.")).toBe("Please try again.");
  });
});

describe("isRequestCurrent", () => {
  it("accepts the latest non-aborted request", () => {
    expect(isRequestCurrent(3, 3, new AbortController().signal)).toBe(true);
  });

  it("rejects stale request IDs", () => {
    expect(isRequestCurrent(2, 3, new AbortController().signal)).toBe(false);
  });

  it("rejects an aborted request even when its ID matches", () => {
    const controller = new AbortController();
    controller.abort();
    expect(isRequestCurrent(3, 3, controller.signal)).toBe(false);
  });
});

describe("isHealthCheckCurrent", () => {
  it("accepts the newest health check without an abort signal", () => {
    expect(isHealthCheckCurrent(4, 4)).toBe(true);
  });

  it("rejects an older health check", () => {
    expect(isHealthCheckCurrent(3, 4)).toBe(false);
  });

  it("accepts the newest non-aborted health check", () => {
    expect(isHealthCheckCurrent(4, 4, new AbortController().signal)).toBe(true);
  });

  it("rejects an aborted health check even when its ID matches", () => {
    const controller = new AbortController();
    controller.abort();
    expect(isHealthCheckCurrent(4, 4, controller.signal)).toBe(false);
  });
});

describe("healthStatusFromPayload", () => {
  it("reports ready only when the provider is configured", () => {
    expect(healthStatusFromPayload({ status: "ok", provider_configured: true })).toBe("ready");
  });

  it("distinguishes a reachable API with a missing provider key", () => {
    expect(healthStatusFromPayload({ status: "ok", provider_configured: false })).toBe("needs-config");
  });

  it.each([
    null,
    [],
    {},
    { status: "offline", provider_configured: true },
    { status: "ok" },
    { status: "ok", provider_configured: "true" },
    { status: "ok", provider_configured: null },
  ])("rejects malformed health payloads: %j", (payload) => {
    expect(healthStatusFromPayload(payload)).toBeNull();
  });
});
