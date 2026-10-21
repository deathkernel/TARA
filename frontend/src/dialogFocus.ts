export type FocusTrapAction = "first" | "last" | "container" | null;
export function focusTrapAction(activeIndex: number, count: number, shiftKey: boolean): FocusTrapAction {
  if (count <= 0) return "container";
  if (activeIndex < 0 || activeIndex >= count) return shiftKey ? "last" : "first";
  if (!shiftKey && activeIndex === count - 1) return "first";
  if (shiftKey && activeIndex === 0) return "last";
  return null;
}
