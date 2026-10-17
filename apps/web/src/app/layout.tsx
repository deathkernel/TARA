import type { Metadata, Viewport } from "next";
import "./globals.css";
export const metadata: Metadata = { title: "TARA — Your thinking space", description: "A focused chat workspace powered by TARA." };
export const viewport: Viewport = { width: "device-width", initialScale: 1, themeColor: "#111318" };
export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) { return <html lang="en"><body>{children}</body></html>; }
