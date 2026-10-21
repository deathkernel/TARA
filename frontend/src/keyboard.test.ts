import {describe,expect,it} from "vitest";
import {isNewChatShortcut,shouldHandleEscape,shouldSendOnEnter,textareaHeight} from "./keyboard";
describe("composer keyboard rules",()=>{
 it("supports Ctrl/Command+K",()=>expect(isNewChatShortcut({key:"k",repeat:false,isComposing:false,metaKey:false,ctrlKey:true})).toBe(true));
 it("does not trigger in editable element",()=>expect(isNewChatShortcut({key:"k",repeat:false,isComposing:false,metaKey:false,ctrlKey:true,targetIsEditable:true})).toBe(false));
 it("supports Enter and Shift+Enter",()=>{expect(shouldSendOnEnter("Enter",false,false)).toBe(true);expect(shouldSendOnEnter("Enter",true,false)).toBe(false)});
 it("does not handle Escape during composition",()=>expect(shouldHandleEscape("Escape",true)).toBe(false));
 it("caps textarea height",()=>expect(textareaHeight(240,180)).toBe(180));
});
