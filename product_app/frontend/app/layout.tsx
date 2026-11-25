import React from "react";
import "./globals.css";

export const metadata = {
  title: "INTI-3K · Centro de Comando Microclimático Cordillera Blanca",
  description:
    "Downscaling climático 27 km → 3 km con IA (EnsembleDownscaler) para gestión de heladas, incendios forestales y seguridad en alta montaña en Áncash.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="es">
      <body className="antialiased selection:bg-sky-500/30 selection:text-sky-100">
        {children}
      </body>
    </html>
  );
}
