/** Small shared pieces: panels, labels, buttons, spinner. Styling lives here
 *  so feature components stay about behaviour. */

import type { ButtonHTMLAttributes, ReactNode } from "react";

export function Panel({
  children,
  className = "",
  label,
}: {
  children: ReactNode;
  className?: string;
  /** Names the section as a landmark, so it can be referred to unambiguously. */
  label?: string;
}) {
  return (
    <section aria-label={label} className={`rounded-lg border border-line bg-panel ${className}`}>
      {children}
    </section>
  );
}

export function Label({ children }: { children: ReactNode }) {
  return <p className="label">{children}</p>;
}

type ButtonProps = ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: "primary" | "ghost";
  children: ReactNode;
};

export function Button({ variant = "ghost", className = "", children, ...props }: ButtonProps) {
  const base =
    "inline-flex items-center justify-center gap-2 rounded-md px-4 py-2 text-sm font-medium " +
    "transition-colors duration-150 disabled:cursor-not-allowed disabled:opacity-40 " +
    "focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent-cool";
  const styles =
    variant === "primary"
      ? "bg-ink text-canvas hover:bg-white"
      : "border border-line-strong bg-raised text-ink hover:border-faint";
  return (
    <button className={`${base} ${styles} ${className}`} {...props}>
      {children}
    </button>
  );
}

export function Spinner({ label }: { label?: string }) {
  return (
    <span className="inline-flex items-center gap-2 text-sm text-muted">
      <span
        aria-hidden="true"
        className="size-3.5 animate-spin rounded-full border-2 border-line-strong border-t-accent"
      />
      {label}
    </span>
  );
}

export function Notice({ tone, children }: { tone: "error" | "info"; children: ReactNode }) {
  const styles =
    tone === "error"
      ? "border-danger/40 bg-danger/10 text-danger"
      : "border-line bg-raised text-muted";
  return (
    <p role={tone === "error" ? "alert" : undefined} className={`rounded-md border px-3 py-2 text-sm ${styles}`}>
      {children}
    </p>
  );
}
