import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  metadataBase: new URL(
    process.env.NEXT_PUBLIC_APP_URL || "http://localhost:3000",
  ),
  title: "PaperPulse — Personal research intelligence",
  description: "Find what matters in your daily research feed.",
  openGraph: {
    title: "PaperPulse",
    description: "Find what matters in your daily research feed.",
    type: "website",
    images: [{ url: "/og.png", width: 1760, height: 917, alt: "PaperPulse research intelligence" }],
  },
  twitter: {
    card: "summary_large_image",
    title: "PaperPulse",
    description: "Find what matters in your daily research feed.",
    images: ["/og.png"],
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      >
        {children}
      </body>
    </html>
  );
}
