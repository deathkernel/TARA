import { describe, expect, it } from "vitest";
import { focusTrapAction } from "./dialogFocus";

describe("focusTrapAction", () => {
  it("wraps forward Tab from the last focusable element to the first", () => {
    expect(focusTrapAction(3, 4, false)).toBe("first");
  });

  it("wraps Shift+Tab from the first focusable element to the last", () => {
    expect(focusTrapAction(0, 4, true)).toBe("last");
  });

  it("keeps focus unchanged when it is between the boundaries", () => {
    expect(focusTrapAction(1, 4, false)).toBeNull();
    expect(focusTrapAction(2, 4, true)).toBeNull();
  });

  it.each([
    ["outside modal with Tab", -1, false, "first"],
    ["outside modal with Shift+Tab", -1, true, "last"],
    ["invalid index with Tab", 5, false, "first"],
    ["invalid index with Shift+Tab", 5, true, "last"],
  ] as const)("returns a safe boundary action for %s", (_label, index, shift, expected) => {
    expect(focusTrapAction(index, 4, shift)).toBe(expected);
  });

  it("returns container when there are no focusable elements", () => {
    expect(focusTrapAction(-1, 0, false)).toBe("container");
    expect(focusTrapAction(0, -1, true)).toBe("container");
  });

  it("does not wrap forward Tab before the last element", () => {
    expect(focusTrapAction(0, 4, false)).toBeNull();
  });

  it("does not wrap backward Tab after the first element", () => {
    expect(focusTrapAction(3, 4, true)).toBeNull();
  });
});
