export type ChatRole = "user" | "assistant";
export type ApiMessage = { role: ChatRole; content: string };

export const MAX_MESSAGE_CHARS = 4000;
export const MAX_CONTEXT_CHARS = 16000;
export const MAX_CONTEXT_MESSAGES = 20;

/**
 * Build a valid bounded transcript for the current backend API.
 * Keep the newest contiguous context that fits, always preserving the latest user message.
 */
export function buildApiMessages(
  history: readonly ApiMessage[],
  latestContent: string,
): ApiMessage[] {
  const latest = latestContent.trim();
  if (!latest) throw new Error("Message content cannot be empty.");
  if (latest.length > MAX_MESSAGE_CHARS) {
    throw new Error(`Each message must be ${MAX_MESSAGE_CHARS} characters or fewer.`);
  }

  let remainingChars = MAX_CONTEXT_CHARS - latest.length;
  const selected: ApiMessage[] = [];
  const historyLimit = MAX_CONTEXT_MESSAGES - 1;

  for (let index = history.length - 1; index >= 0 && selected.length < historyLimit; index -= 1) {
    const item = history[index];
    if (!item || (item.role !== "user" && item.role !== "assistant")) break;
    const content = typeof item.content === "string" ? item.content.trim() : "";
    if (!content || content.length > MAX_MESSAGE_CHARS || content.length > remainingChars) break;

    selected.push({ role: item.role, content });
    remainingChars -= content.length;
  }

  selected.reverse();
  selected.push({ role: "user", content: latest });
  return selected;
}
