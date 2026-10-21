export type ShortcutKeyEvent = { key:string; repeat:boolean; isComposing:boolean; metaKey:boolean; ctrlKey:boolean; targetIsEditable?:boolean };
export function isNewChatShortcut(event: ShortcutKeyEvent): boolean { return event.key.toLowerCase()==="k" && (event.metaKey||event.ctrlKey) && !event.repeat && !event.isComposing && !event.targetIsEditable; }
export function shouldHandleEscape(key:string,isComposing:boolean): boolean { return key==="Escape"&&!isComposing; }
export function shouldSendOnEnter(key:string,shiftKey:boolean,isComposing:boolean): boolean { return key==="Enter"&&!shiftKey&&!isComposing; }
export function textareaHeight(height:number,max=180):number { if(!Number.isFinite(height)||height<0)return 0;return Number.isFinite(max)&&max>0?Math.min(height,max):height; }
