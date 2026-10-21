import { describe, expect, it } from "vitest";
import { assistantMessageFromPayload, apiErrorFromPayload, errorMessageFromUnknown, healthStatusFromPayload, isHealthCheckCurrent, isRequestCurrent } from "./chatApi";
describe("API response helpers",()=>{
 it("reads a valid answer",()=>expect(assistantMessageFromPayload({message:" Hi "})).toBe("Hi"));
 it("rejects empty answer",()=>expect(()=>assistantMessageFromPayload({message:" "})).toThrow());
 it("formats server validation errors",()=>expect(apiErrorFromPayload(400,{detail:"Invalid message"})).toBe("Invalid message"));
 it("falls back to HTTP status",()=>expect(apiErrorFromPayload(503,{})).toBe("TARA backend returned HTTP 503."));
 it("normalizes unknown errors",()=>expect(errorMessageFromUnknown(null)).toBe("An unexpected error occurred."));
 it("checks response freshness",()=>expect(isRequestCurrent(2,2,new AbortController().signal)).toBe(true));
 it("ignores stale health result",()=>expect(isHealthCheckCurrent(1,2)).toBe(false));
 it("validates backend status",()=>expect(healthStatusFromPayload({status:"ok",provider_configured:true})).toBe("ready"));
});
