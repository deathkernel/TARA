export type ShortcutKeyEvent = {
  key: string;
  repeat: boolean;
  isComposing: boolean;
  metaKey: boolean;
  ctrlKey: boolean;
};

export function isNewChatShortcut(event: ShortcutKeyEvent): boolean {
  return !event.repeat
    && !event.isComposing
    && (event.metaKey || event.ctrlKey)
    && event.key.toLowerCase() === "k";
}

export function shouldSendOnEnter(key: string, shiftKey: boolean, isComposing: boolean): boolean {
  return key === "Enter" && !shiftKey && !isComposing;
}
