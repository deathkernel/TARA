export type ShortcutKeyEvent = {
  key: string;
  repeat: boolean;
  isComposing: boolean;
  metaKey: boolean;
  ctrlKey: boolean;
  targetIsEditable?: boolean;
};

export function isNewChatShortcut(event: ShortcutKeyEvent): boolean {
  return !event.repeat
    && !event.isComposing
    && !event.targetIsEditable
    && (event.metaKey || event.ctrlKey)
    && event.key.toLowerCase() === "k";
}

export function shouldSendOnEnter(key: string, shiftKey: boolean, isComposing: boolean): boolean {
  return key === "Enter" && !shiftKey && !isComposing;
}

/** Ignore Escape while an IME is composing text; otherwise it can cancel input. */
export function shouldHandleEscape(key: string, isComposing: boolean): boolean {
  return key === "Escape" && !isComposing;
}

/** Clamp textarea height to a sensible range, including malformed layout measurements. */
export function textareaHeight(scrollHeight: number, maxHeight = 180): number {
  if (!Number.isFinite(scrollHeight) || scrollHeight < 0) return 0;
  if (!Number.isFinite(maxHeight) || maxHeight < 0) return scrollHeight;
  return Math.min(scrollHeight, maxHeight);
}
