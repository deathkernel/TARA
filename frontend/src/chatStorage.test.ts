import {describe,expect,it} from "vitest";
import {isConversation,isMessage,parseHistory} from "./chatStorage";
const msg={id:"m1",role:"user",content:"Hello"} as const;
const chat={id:"c1",title:"Hello",updatedAt:123,messages:[msg]};
describe("local chat history validation",()=>{
 it("accepts valid messages and conversations",()=>{expect(isMessage(msg)).toBe(true);expect(isConversation(chat)).toBe(true)});
 it("rejects empty IDs and blank content",()=>{expect(isMessage({...msg,id:" "})).toBe(false);expect(isMessage({...msg,content:" "})).toBe(false)});
 it("handles corrupt history",()=>expect(parseHistory("{")).toEqual([]));
 it("caps history",()=>expect(parseHistory(JSON.stringify(Array.from({length:35},(_,i)=>({...chat,id:String(i)}))))).toHaveLength(30));
});
