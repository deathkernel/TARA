import {describe,expect,it} from "vitest";
import {focusTrapAction} from "./dialogFocus";
describe("modal focus boundaries",()=>{
 it("wraps forward focus",()=>expect(focusTrapAction(3,4,false)).toBe("first"));
 it("wraps backward focus",()=>expect(focusTrapAction(0,4,true)).toBe("last"));
 it("does not trap middle tabs",()=>expect(focusTrapAction(1,4,false)).toBeNull());
 it("handles no controls",()=>expect(focusTrapAction(-1,0,false)).toBe("container"));
});
