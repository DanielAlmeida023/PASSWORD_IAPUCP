import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        ops: {
          bg: "#050911",
          panel: "rgba(9, 15, 28, 0.88)",
          border: "rgba(56, 189, 248, 0.22)",
        },
      },
    },
  },
  plugins: [],
};

export default config;
