export type FocusTrapAction = "first" | "last" | "container" | null;

/**
 * Decide how a modal should handle Tab based on the active focusable index.
 * Use -1 for focus outside the modal or when the active element is not focusable.
 */
export function focusTrapAction(
  activeIndex: number,
  focusableCount: number,
  shiftKey: boolean,
): FocusTrapAction {
  if (focusableCount <= 0) return "container";
  if (activeIndex < 0 || activeIndex >= focusableCount) return shiftKey ? "last" : "first";
  if (shiftKey && activeIndex === 0) return "last";
  if (!shiftKey && activeIndex === focusableCount - 1) return "first";
  return null;
}
