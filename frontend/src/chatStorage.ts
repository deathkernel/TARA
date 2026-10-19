export type Message = { id: string; role: "user" | "assistant"; content: string };
export type Conversation = { id: string; title: string; updatedAt: number; messages: Message[] };

export const HISTORY_KEY = "tara.chat.history.v1";
export const THEME_KEY = "tara.chat.theme.v1";

export function isMessage(value: unknown): value is Message {
  if (typeof value !== "object" || value === null) return false;
  const message = value as Record<string, unknown>;
  return typeof message.id === "string" && message.id.trim().length > 0 && (message.role === "user" || message.role === "assistant") && typeof message.content === "string";
}

export function isConversation(value: unknown): value is Conversation {
  if (typeof value !== "object" || value === null) return false;
  const chat = value as Record<string, unknown>;
  return typeof chat.id === "string" && chat.id.trim().length > 0 && typeof chat.title === "string" && chat.title.trim().length > 0 && typeof chat.updatedAt === "number" && Number.isFinite(chat.updatedAt) && Array.isArray(chat.messages) && chat.messages.length > 0 && chat.messages.every(isMessage);
}

export function parseHistory(raw: string | null): Conversation[] {
  try {
    const value: unknown = JSON.parse(raw ?? "[]");
    if (!Array.isArray(value)) return [];
    return value.filter(isConversation).slice(0, 30);
  } catch {
    return [];
  }
}
