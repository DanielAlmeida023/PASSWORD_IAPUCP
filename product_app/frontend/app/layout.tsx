import React from "react";

export const metadata = {
  title: "Hackathon IA PUCP 2026 — Producto Inteligente",
  description: "Aplicativo funcional con ingesta de datos actuales y modelo de IA entrenado.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="es">
      <body style={{ margin: 0 }}>{children}</body>
    </html>
  );
}
