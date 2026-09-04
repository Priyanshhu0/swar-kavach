/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        void: "#0B1220",
        panel: "#111A2E",
        panel2: "#16213A",
        panel3: "#1C2A47",
        edge: "rgba(140,160,196,0.14)",
        cyan: {
          DEFAULT: "#35D0BA",
          soft: "#2AA890",
        },
        blue: {
          DEFAULT: "#4C8DFF",
          soft: "#3A6FD1",
        },
        amber: "#F5B942",
        coral: "#FF5470",
        ink: "#EAF2FF",
        muted: "#8CA0C4",
        faint: "#5C7096",
      },
      fontFamily: {
        display: ["'Space Grotesk'", "sans-serif"],
        body: ["'Inter'", "sans-serif"],
        mono: ["'IBM Plex Mono'", "monospace"],
      },
      boxShadow: {
        glow: "0 0 0 1px rgba(53,208,186,0.25), 0 0 24px rgba(53,208,186,0.12)",
        panel: "0 1px 0 rgba(255,255,255,0.03) inset, 0 8px 24px rgba(0,0,0,0.35)",
      },
      keyframes: {
        scan: {
          "0%": { transform: "translateY(-100%)" },
          "100%": { transform: "translateY(100%)" },
        },
        pulseRing: {
          "0%": { boxShadow: "0 0 0 0 rgba(53,208,186,0.35)" },
          "100%": { boxShadow: "0 0 0 14px rgba(53,208,186,0)" },
        },
      },
      animation: {
        scan: "scan 2.4s linear infinite",
        pulseRing: "pulseRing 1.6s ease-out infinite",
      },
    },
  },
  plugins: [],
};
