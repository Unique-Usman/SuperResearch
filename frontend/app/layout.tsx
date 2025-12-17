import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import Link from "next/link";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "Predli Research Agent",
  description: "Advanced AI-powered research with parallel search and RAG",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <nav className="fixed z-50 w-full border-b border-zinc-800 bg-black/50 backdrop-blur-sm">
          <div className="container flex justify-between items-center py-4 px-6 mx-auto">
            <Link href="/" className="flex gap-2 items-center">
              <div className="w-8 h-8 rounded-lg bg-accent"></div>
              <span className="text-xl font-bold">SuperResearch</span>
            </Link>
            <Link href="/research">
              <button className="btn-accent">Start Research</button>
            </Link>
          </div>
        </nav>
        <main className="pt-20">{children}</main>
        <footer className="py-8 mt-20 border-t border-zinc-800">
          <div className="container px-6 mx-auto text-center text-zinc-400">
            <p>
              © 2025 Predli Research Agent. Built with Next.js & LangGraph.
            </p>
          </div>
        </footer>
      </body>
    </html>
  );
}
