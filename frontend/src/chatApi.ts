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
  if (typeof payload === "object" && payload !== null && "detail" in payload) {
    const detail = (payload as { detail?: unknown }).detail;
    if (typeof detail === "string" && detail.trim()) return detail.trim();
  }
  return `TARA backend returned HTTP ${status}.`;
}

export function assistantMessageFromPayload(payload: unknown): string {
  if (typeof payload !== "object" || payload === null || !("message" in payload) || typeof (payload as { message?: unknown }).message !== "string") {
    throw new Error("The backend response did not contain a valid message.");
  }
  const message = ((payload as { message: string }).message).trim();
  if (!message) throw new Error("The backend returned an empty message. Please retry.");
  return message;
}

/**
 * Convert the health endpoint contract into a UI status.
 * A malformed payload is rejected instead of being mistaken for a configured API.
 */
export function healthStatusFromPayload(payload: unknown): "ready" | "needs-config" | null {
  if (typeof payload !== "object" || payload === null) return null;
  if (!("status" in payload) || !("provider_configured" in payload)) return null;
  const health = payload as { status?: unknown; provider_configured?: unknown };
  if (health.status !== "ok" || typeof health.provider_configured !== "boolean") return null;
  return health.provider_configured ? "ready" : "needs-config";
}
