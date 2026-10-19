export function isRequestCurrent(requestId: number, currentRequestId: number, signal: AbortSignal): boolean {
  return requestId === currentRequestId && !signal.aborted;
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
