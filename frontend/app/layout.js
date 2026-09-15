import { Bebas_Neue, Inter } from "next/font/google";
import "./globals.css";

const displayFont = Bebas_Neue({
  weight: "400",
  subsets: ["latin"],
  variable: "--font-display",
});

const bodyFont = Inter({
  subsets: ["latin"],
  variable: "--font-body",
});

export const metadata = {
  title: "Chatbot Copa do Mundo",
  description: "Chatbot com RAG sobre a história da Copa do Mundo FIFA",
};

export default function RootLayout({ children }) {
  return (
    <html lang="pt-BR">
      <body className={`${displayFont.variable} ${bodyFont.variable}`}>
        {children}
      </body>
    </html>
  );
}
