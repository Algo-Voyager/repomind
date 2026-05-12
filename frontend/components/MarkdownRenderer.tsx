"use client";

import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeRaw from "rehype-raw";
import type { Components } from "react-markdown";

const BTN =
  "flex items-center justify-center rounded-md bg-muted border border-border text-foreground hover:bg-accent transition-colors text-sm select-none";

function MermaidBlock({ code }: { code: string }) {
  const ref = useRef<HTMLDivElement>(null);
  const [svgContent, setSvgContent] = useState("");
  const [isOpen, setIsOpen] = useState(false);
  const [zoom, setZoom] = useState(1);

  useEffect(() => {
    let cancelled = false;
    import("mermaid").then((m) => {
      if (cancelled || !ref.current) return;
      const mermaid = m.default;
      mermaid.initialize({
        startOnLoad: false,
        theme: "dark",
        themeVariables: {
          background:           "#0d0d10",
          primaryColor:         "#0c4a6e",
          primaryBorderColor:   "#0ea5e9",
          primaryTextColor:     "#f0f9ff",
          lineColor:            "#38bdf8",
          secondaryColor:       "#164e63",
          tertiaryColor:        "#1e3a5f",
          edgeLabelBackground:  "#0d0d10",
          nodeTextColor:        "#f0f9ff",
          clusterBkg:           "#111827",
          titleColor:           "#bae6fd",
        },
      });
      const id = `mermaid-${Math.random().toString(36).slice(2)}`;
      mermaid
        .render(id, code)
        .then(({ svg }) => {
          if (cancelled || !ref.current) return;
          ref.current.innerHTML = svg;
          const svgEl = ref.current.querySelector("svg");
          if (svgEl) {
            svgEl.removeAttribute("width");
            svgEl.removeAttribute("height");
            svgEl.style.width = "100%";
            svgEl.style.height = "auto";
            svgEl.style.maxWidth = "100%";
          }
          setSvgContent(ref.current.innerHTML);
        })
        .catch(() => {
          if (!cancelled && ref.current) {
            ref.current.innerHTML = `<pre class="text-xs text-muted-foreground p-2 whitespace-pre-wrap">${code}</pre>`;
          }
        });
    });
    return () => { cancelled = true; };
  }, [code]);

  const open = () => { if (svgContent) { setZoom(1); setIsOpen(true); } };
  const close = () => setIsOpen(false);
  const zoomIn  = () => setZoom(z => Math.min(4, +(z * 1.25).toFixed(2)));
  const zoomOut = () => setZoom(z => Math.max(0.25, +(z * 0.8).toFixed(2)));
  const reset   = () => setZoom(1);

  const handleWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    setZoom(z => Math.min(4, Math.max(0.25, z * (e.deltaY > 0 ? 0.85 : 1.18))));
  };

  return (
    <>
      {/* Inline preview — click to open lightbox */}
      <div className="relative group my-4">
        <div
          ref={ref}
          className="w-full rounded-xl border border-border bg-[#0d0d10] p-6 overflow-x-auto cursor-zoom-in hover:border-sky-500/50 transition-colors"
          onClick={open}
        />
        {svgContent && (
          <span className="absolute top-2 right-2 text-[10px] text-muted-foreground bg-black/70 px-2 py-0.5 rounded opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none">
            Click to zoom
          </span>
        )}
      </div>

      {/* Lightbox */}
      {isOpen && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/85 backdrop-blur-sm"
          onClick={close}
        >
          <div
            className="relative flex flex-col rounded-xl border border-border bg-[#0d0d10]"
            style={{ width: "90vw", height: "90vh" }}
            onClick={e => e.stopPropagation()}
          >
            {/* Toolbar */}
            <div className="flex items-center justify-between px-4 py-2 border-b border-border shrink-0">
              <span className="text-xs text-muted-foreground">Scroll to zoom · drag scrollbars to pan</span>
              <div className="flex items-center gap-1.5">
                <button onClick={zoomOut} className={`${BTN} w-8 h-8`} title="Zoom out">−</button>
                <span className="text-xs text-muted-foreground w-12 text-center tabular-nums">
                  {Math.round(zoom * 100)}%
                </span>
                <button onClick={zoomIn}  className={`${BTN} w-8 h-8`} title="Zoom in">+</button>
                <button onClick={reset}   className={`${BTN} px-2 h-8 text-xs`} title="Reset zoom">Reset</button>
                <button onClick={close}   className={`${BTN} w-8 h-8`} title="Close">✕</button>
              </div>
            </div>

            {/* Scrollable diagram area */}
            <div className="overflow-auto flex-1 p-6" onWheel={handleWheel}>
              <div
                style={{ width: `${zoom * 100}%`, minWidth: "100%", margin: "0 auto" }}
                dangerouslySetInnerHTML={{ __html: svgContent }}
              />
            </div>
          </div>
        </div>
      )}
    </>
  );
}

const components: Components = {
  // Headings
  h1: ({ children }) => (
    <h1 className="text-xl font-bold mt-4 mb-2 text-foreground border-b border-border pb-1">
      {children}
    </h1>
  ),
  h2: ({ children }) => (
    <h2 className="text-lg font-semibold mt-4 mb-2 text-foreground">{children}</h2>
  ),
  h3: ({ children }) => (
    <h3 className="text-base font-semibold mt-3 mb-1 text-foreground">{children}</h3>
  ),
  // Paragraphs
  p: ({ children }) => (
    <p className="mb-2 last:mb-0 leading-relaxed">{children}</p>
  ),
  // Lists — use ml-5 not list-inside to prevent number/bullet wrapping onto own line
  ul: ({ children }) => (
    <ul className="list-disc ml-5 space-y-1 mb-3">{children}</ul>
  ),
  ol: ({ children }) => (
    <ol className="list-decimal ml-5 space-y-1 mb-3">{children}</ol>
  ),
  li: ({ children }) => <li className="leading-relaxed pl-1">{children}</li>,
  // Inline code
  code: ({ className, children, ...props }) => {
    const lang = (className ?? "").replace("language-", "");
    const isBlock = !!className;
    const code = String(children).replace(/\n$/, "");

    if (isBlock && lang === "mermaid") {
      return <MermaidBlock code={code} />;
    }
    if (isBlock) {
      return (
        <pre className="my-2 p-3 rounded-md bg-muted text-xs overflow-x-auto border border-border">
          <code className="text-foreground/90">{children}</code>
        </pre>
      );
    }
    return (
      <code
        className="px-1.5 py-0.5 rounded bg-muted text-xs font-mono text-foreground/90 border border-border"
        {...props}
      >
        {children}
      </code>
    );
  },
  // Blockquote
  blockquote: ({ children }) => (
    <blockquote className="border-l-2 border-primary pl-3 my-2 text-muted-foreground italic">
      {children}
    </blockquote>
  ),
  // Table
  table: ({ children }) => (
    <div className="overflow-x-auto my-3">
      <table className="w-full text-xs border border-border rounded-md overflow-hidden">
        {children}
      </table>
    </div>
  ),
  thead: ({ children }) => (
    <thead className="bg-muted text-muted-foreground">{children}</thead>
  ),
  tbody: ({ children }) => (
    <tbody className="divide-y divide-border">{children}</tbody>
  ),
  tr: ({ children }) => (
    <tr className="hover:bg-accent/30 transition-colors">{children}</tr>
  ),
  th: ({ children }) => (
    <th className="px-3 py-2 text-left font-medium">{children}</th>
  ),
  td: ({ children }) => <td className="px-3 py-2">{children}</td>,
  // Horizontal rule
  hr: () => <hr className="border-border my-4" />,
  // Strong / em
  strong: ({ children }) => (
    <strong className="font-semibold text-foreground">{children}</strong>
  ),
  em: ({ children }) => <em className="italic text-foreground/80">{children}</em>,
  // Links
  a: ({ href, children }) => (
    <a
      href={href}
      target="_blank"
      rel="noreferrer"
      className="text-primary underline underline-offset-2 hover:text-primary/80"
    >
      {children}
    </a>
  ),
};

export function MarkdownRenderer({ content }: { content: string }) {
  return (
    <div className="prose-sm max-w-none text-sm text-foreground">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        rehypePlugins={[rehypeRaw]}
        components={components}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}
