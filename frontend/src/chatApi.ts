export function isRequestCurrent(requestId: number, currentRequestId: number, signal: AbortSignal): boolean {
  return requestId === currentRequestId && !signal.aborted;
}

/** Reject health-check results from an older check or an aborted component lifecycle. */
export function isHealthCheckCurrent(checkId: number, currentCheckId: number, signal?: AbortSignal): boolean {
  return checkId === currentCheckId && !signal?.aborted;
}

/** Convert unknown thrown values into a safe, non-empty message for the UI. */
export function errorMessageFromUnknown(error: unknown, fallback = "An unexpected error occurred."): string {
  if (error instanceof Error && error.message.trim()) return error.message.trim();
  if (typeof error === "string" && error.trim()) return error.trim();
  return fallback;
}

export function apiErrorFromPayload(status: number, payload: unknown): string {
  if (typeof payload === "object" && payload !== null && ("detail" in payload || "error" in payload)) {
    const raw = payload as { detail?: unknown; error?: unknown };
    const detail = raw.detail ?? raw.error;
    if (typeof detail === "string" && detail.trim()) return detail.trim();
    // FastAPI validation errors commonly return an array of { loc, msg, type } objects.
    if (Array.isArray(detail)) {
      const messages = detail
        .map((item) => {
          if (typeof item !== "object" || item === null || !("msg" in item)) return "";
          const message = (item as { msg?: unknown }).msg;
          return typeof message === "string" ? message.trim() : "";
        })
        .filter(Boolean)
        .slice(0, 3);
      if (messages.length) return messages.join(" ");
    }
  }
  return `TARA backend returned HTTP ${status}.`;
}

export function assistantMessageFromPayload(payload: unknown): string {
  if (typeof payload !== "object" || payload === null) throw new Error("The backend response did not contain a valid message.");
  const raw = payload as { message?: unknown; reply?: unknown };
  const text = typeof raw.message === "string" ? raw.message : raw.reply;
  if (typeof text !== "string") throw new Error("The backend response did not contain a valid message.");
  const message = text.trim();
  if (!message) throw new Error("The backend returned an empty message. Please retry.");
  return message;
}

/**
 * Convert the health endpoint contract into a UI status.
 * A malformed payload is rejected instead of being mistaken for a configured API.
 */
export function healthStatusFromPayload(payload: unknown): "ready" | "needs-config" | "offline" | null {
  if (typeof payload !== "object" || payload === null) return null;
  if (!("ok" in payload) || !("ollama" in payload) || !("model_installed" in payload)) return null;
  const health = payload as { ok?: unknown; ollama?: unknown; model_installed?: unknown };
  if (health.ok === true && health.model_installed === true) return "ready";
  if (health.ollama === "disconnected") return "offline";
  if (health.model_installed === false) return "needs-config";
  return null;
}
