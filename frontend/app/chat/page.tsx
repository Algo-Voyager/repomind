"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Paperclip, SendIcon, LoaderIcon, Command, ArrowUpIcon, Copy, Check } from "lucide-react";
import { triggerQuery, pollResult, fetchSessionLogs, type AgentResult, type LogEntry, type ContextMessage } from "@/lib/api";
import { ReasoningExpander } from "@/components/ReasoningExpander";
import { MarkdownRenderer } from "@/components/MarkdownRenderer";
import {
  useAutoResizeTextarea,
  AnimatedTextarea,
  TypingDots,
} from "@/components/ui/animated-ai-chat";
import { cn } from "@/lib/utils";

interface Message {
  role: "user" | "assistant";
  content: string;
  steps?: number;
  logs?: LogEntry[];
  error?: boolean;
}

type ChatHistories = Record<string, Message[]>;

const HINT_QUESTIONS = [
  "How is authentication handled?",
  "Explain the data ingestion pipeline",
  "Where are API routes defined?",
];

function BotAvatar() {
  return (
    <div className="flex-shrink-0 w-7 h-7 rounded-full bg-sky-500/20 border border-sky-500/30 flex items-center justify-center text-xs font-bold text-sky-300">
      KW
    </div>
  );
}

export default function ChatPage() {
  const [collection, setCollection] = useState<string>("");
  const [histories, setHistories] = useState<ChatHistories>({});
  const [input, setInput] = useState("");
  const bottomRef = useRef<HTMLDivElement>(null);
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });
  const [inputFocused, setInputFocused] = useState(false);

  const { textareaRef, adjustHeight } = useAutoResizeTextarea({
    minHeight: 44,
    maxHeight: 144,
  });

  // Per-collection concurrency & queue
  const inFlightRef = useRef<Record<string, boolean>>({});
  const queuesRef = useRef<Record<string, string[]>>({});
  const [loadingCols, setLoadingCols] = useState<Record<string, boolean>>({});
  const [queueCounts, setQueueCounts] = useState<Record<string, number>>({});

  // Per-collection compressed context (not shown in UI)
  const contextRef = useRef<Record<string, ContextMessage[]>>({});

  // ── Persistence (localStorage) ───────────────────────────────────────
  // Survives page refresh AND tab close. Stored under one key so we save
  // histories + the (server-compressed) context together. Bump the version
  // suffix on the key if the shape of stored state changes.
  const CHAT_STATE_KEY = "repomind:chat-state:v1";
  const hydratedRef = useRef(false);

  // Load once on mount BEFORE the first save effect can clobber storage.
  useEffect(() => {
    try {
      const raw = window.localStorage.getItem(CHAT_STATE_KEY);
      if (raw) {
        const saved = JSON.parse(raw) as {
          histories?: ChatHistories;
          contexts?: Record<string, ContextMessage[]>;
        };
        if (saved.histories && typeof saved.histories === "object") {
          setHistories(saved.histories);
        }
        if (saved.contexts && typeof saved.contexts === "object") {
          contextRef.current = saved.contexts;
        }
      }
    } catch { /* private mode / quota / parse error — silently start fresh */ }
    hydratedRef.current = true;
  }, []);

  // Save on every histories change. contextRef updates happen in the same
  // request-completion code path as histories updates, so this catches both.
  useEffect(() => {
    if (!hydratedRef.current) return;  // skip the initial empty {} write before hydrate
    try {
      window.localStorage.setItem(
        CHAT_STATE_KEY,
        JSON.stringify({ histories, contexts: contextRef.current }),
      );
    } catch { /* quota exceeded — drop save silently */ }
  }, [histories]);

  // Mouse tracking for gradient follow effect
  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) =>
      setMousePosition({ x: e.clientX, y: e.clientY });
    window.addEventListener("mousemove", handleMouseMove);
    return () => window.removeEventListener("mousemove", handleMouseMove);
  }, []);

  useEffect(() => {
    function onCollectionChanged(e: Event) {
      const name = (e as CustomEvent<string>).detail;
      setCollection(name);
    }
    window.addEventListener("collection-changed", onCollectionChanged);
    const stored = sessionStorage.getItem("selected-collection");
    if (stored) setCollection(stored);
    return () => window.removeEventListener("collection-changed", onCollectionChanged);
  }, []);

  useEffect(() => {
    function onCollectionChanged(e: Event) {
      const name = (e as CustomEvent<string>).detail;
      sessionStorage.setItem("selected-collection", name);
    }
    window.addEventListener("collection-changed", onCollectionChanged);
    return () => window.removeEventListener("collection-changed", onCollectionChanged);
  }, []);

  const messages: Message[] = collection ? (histories[collection] ?? []) : [];
  const isCurrentLoading = !!loadingCols[collection];
  const currentQueueCount = queueCounts[collection] ?? 0;
  const [copiedIndex, setCopiedIndex] = useState<number | null>(null);

  async function handleCopy(i: number, content: string) {
    try {
      await navigator.clipboard.writeText(content);
      setCopiedIndex(i);
      setTimeout(() => setCopiedIndex((cur) => (cur === i ? null : cur)), 1500);
    } catch (e) {
      console.error("Failed to copy message", e);
    }
  }

  function appendMessage(col: string, msg: Message) {
    setHistories((prev) => ({ ...prev, [col]: [...(prev[col] ?? []), msg] }));
  }

  function updateLastAssistant(col: string, updater: (m: Message) => Message) {
    setHistories((prev) => {
      const msgs = [...(prev[col] ?? [])];
      for (let i = msgs.length - 1; i >= 0; i--) {
        if (msgs[i].role === "assistant") {
          msgs[i] = updater(msgs[i]);
          break;
        }
      }
      return { ...prev, [col]: msgs };
    });
  }

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function executeQuery(col: string, prompt: string) {
    const historyToSend: ContextMessage[] = contextRef.current[col] ?? [];
    appendMessage(col, { role: "assistant", content: "…" });

    try {
      const sessionId = await triggerQuery(prompt, col, historyToSend);

      let result: AgentResult | null = null;
      for (let i = 0; i < 120; i++) {
        await new Promise((r) => setTimeout(r, 2000));
        result = await pollResult(sessionId);
        if (result) break;
      }

      if (!result) {
        updateLastAssistant(col, (m) => ({
          ...m,
          content: "The agent did not respond in time. The Modal service may be cold-starting — try again.",
          error: true,
        }));
        return;
      }

      const logs = await fetchSessionLogs(result.session_id);
      updateLastAssistant(col, (m) => ({
        ...m,
        content: result!.answer,
        steps: result!.steps,
        logs,
      }));

      const updatedContext: ContextMessage[] = [
        ...(result.compressed_history ?? historyToSend),
        { role: "user", content: prompt },
        { role: "assistant", content: result.answer },
      ];
      contextRef.current[col] = updatedContext;
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : "Something went wrong. Please try again.";
      updateLastAssistant(col, (m) => ({ ...m, content: msg, error: true }));
    } finally {
      const q = queuesRef.current[col] ?? [];
      const next = q.shift();
      queuesRef.current[col] = q;
      setQueueCounts((prev) => ({ ...prev, [col]: q.length }));
      if (next !== undefined) {
        executeQuery(col, next);
      } else {
        inFlightRef.current[col] = false;
        setLoadingCols((prev) => ({ ...prev, [col]: false }));
      }
    }
  }

  const send = useCallback(() => {
    const prompt = input.trim();
    if (!prompt || !collection) return;
    setInput("");
    adjustHeight(true);
    appendMessage(collection, { role: "user", content: prompt });

    if (inFlightRef.current[collection]) {
      const q = queuesRef.current[collection] ?? [];
      q.push(prompt);
      queuesRef.current[collection] = q;
      setQueueCounts((prev) => ({ ...prev, [collection]: q.length }));
    } else {
      inFlightRef.current[collection] = true;
      setLoadingCols((prev) => ({ ...prev, [collection]: true }));
      executeQuery(collection, prompt);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [input, collection]);

  // ── No collection selected ──────────────────────────────────────────────────
  if (!collection) {
    return (
      <div className="flex flex-col flex-1 items-center justify-center relative overflow-hidden">
        {/* Gradient blobs */}
        <div className="absolute inset-0 pointer-events-none">
          <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-sky-500/10 rounded-full filter blur-[128px] animate-pulse" />
          <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-blue-500/10 rounded-full filter blur-[128px] animate-pulse delay-700" />
        </div>
        <motion.div
          className="relative z-10 text-center space-y-3"
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          <div className="w-14 h-14 rounded-2xl bg-sky-500/10 border border-sky-500/20 flex items-center justify-center text-2xl font-bold text-sky-300 mx-auto">
            KW
          </div>
          <p className="text-base font-medium text-foreground">No repository selected</p>
          <p className="text-sm text-muted-foreground">
            Choose an indexed repo from the sidebar to start chatting.
          </p>
        </motion.div>
      </div>
    );
  }

  // ── Chat view ───────────────────────────────────────────────────────────────
  return (
    <div className="flex flex-col flex-1 min-h-0 relative overflow-hidden">
      {/* Subtle gradient blobs in background */}
      <div className="absolute inset-0 pointer-events-none overflow-hidden">
        <div className="absolute -top-32 -left-32 w-96 h-96 bg-sky-500/5 rounded-full filter blur-[128px]" />
        <div className="absolute -bottom-32 -right-32 w-96 h-96 bg-blue-500/5 rounded-full filter blur-[128px]" />
      </div>

      {/* Header */}
      <motion.div
        className="relative z-10 shrink-0 px-5 py-3 border-b border-border bg-card/60 backdrop-blur-sm flex items-center gap-3"
        initial={{ opacity: 0, y: -8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3 }}
      >
        <motion.div
          className={cn(
            "w-2 h-2 rounded-full shrink-0",
            isCurrentLoading ? "bg-sky-400" : "bg-green-400"
          )}
          animate={isCurrentLoading ? { opacity: [1, 0.4, 1] } : { opacity: 1 }}
          transition={isCurrentLoading ? { duration: 1.2, repeat: Infinity } : {}}
        />
        <div className="min-w-0 flex-1">
          <p className="text-sm font-medium truncate text-foreground">{collection}</p>
          <p className="text-xs text-muted-foreground">
            {isCurrentLoading
              ? currentQueueCount > 0
                ? `Processing · ${currentQueueCount} queued`
                : "Processing…"
              : "Kevin agent"}
          </p>
        </div>
      </motion.div>

      {/* Messages */}
      <div className="relative z-10 flex-1 overflow-y-auto px-4 py-6">
        {messages.length === 0 ? (
          <motion.div
            className="flex flex-col items-center justify-center h-full gap-4 text-center"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.2 }}
          >
            <p className="text-sm text-muted-foreground">Ask anything about the codebase.</p>
            <div className="flex flex-wrap gap-2 justify-center max-w-sm">
              {HINT_QUESTIONS.map((hint) => (
                <motion.button
                  key={hint}
                  onClick={() => {
                    setInput(hint);
                    textareaRef.current?.focus();
                  }}
                  className="text-xs px-3 py-1.5 rounded-full border border-border text-muted-foreground hover:text-foreground hover:border-sky-500/50 hover:bg-sky-500/5 transition-all"
                  whileHover={{ scale: 1.03 }}
                  whileTap={{ scale: 0.97 }}
                >
                  {hint}
                </motion.button>
              ))}
            </div>
          </motion.div>
        ) : (
          <div className="flex flex-col gap-5 max-w-3xl mx-auto w-full">
            <AnimatePresence initial={false}>
              {messages.map((msg, i) => (
                <motion.div
                  key={i}
                  className={`group flex gap-3 ${msg.role === "user" ? "justify-end" : "justify-start"}`}
                  initial={{ opacity: 0, y: 12 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.25, ease: "easeOut" }}
                >
                  {msg.role === "assistant" && <BotAvatar />}

                  <div
                    className={cn(
                      "flex flex-col gap-1",
                      msg.role === "user"
                        ? "items-end max-w-[70%]"
                        : "items-start flex-1 min-w-0"
                    )}
                  >
                    <div
                      className={cn(
                        "rounded-2xl px-4 py-3 text-sm leading-relaxed",
                        msg.role === "user"
                          ? "bg-primary text-primary-foreground rounded-br-sm"
                          : msg.error
                          ? "bg-destructive/10 text-destructive border border-destructive/30 rounded-bl-sm w-full"
                          : "bg-card border border-border text-foreground rounded-bl-sm w-full"
                      )}
                    >
                      {msg.role === "assistant" && msg.content === "…" ? (
                        <div className="flex items-center gap-2">
                          <span className="text-muted-foreground text-xs">Agent working</span>
                          <TypingDots />
                        </div>
                      ) : msg.role === "assistant" ? (
                        <MarkdownRenderer content={msg.content} />
                      ) : (
                        <div className="whitespace-pre-wrap">{msg.content}</div>
                      )}
                    </div>

                    {msg.role === "assistant" &&
                      msg.content !== "…" &&
                      !(isCurrentLoading && i === messages.length - 1) && (
                        <button
                          onClick={() => handleCopy(i, msg.content)}
                          className="opacity-0 group-hover:opacity-100 transition-opacity flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground px-1.5 py-1 rounded-md hover:bg-accent/50"
                          aria-label="Copy message"
                        >
                          {copiedIndex === i ? (
                            <>
                              <Check className="w-3.5 h-3.5" />
                              <span>Copied</span>
                            </>
                          ) : (
                            <Copy className="w-3.5 h-3.5" />
                          )}
                        </button>
                      )}

                    {msg.steps !== undefined && msg.logs !== undefined && (
                      <div className="w-full">
                        <ReasoningExpander steps={msg.steps} logs={msg.logs} />
                      </div>
                    )}
                  </div>
                </motion.div>
              ))}
            </AnimatePresence>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {/* Animated "agent thinking" pill */}
      <AnimatePresence>
        {isCurrentLoading && currentQueueCount === 0 && (
          <motion.div
            className="absolute bottom-24 left-1/2 -translate-x-1/2 z-20 backdrop-blur-xl bg-black/60 rounded-full px-4 py-2 shadow-lg border border-white/10"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 10 }}
          >
            <div className="flex items-center gap-3">
              <div className="w-6 h-6 rounded-full bg-sky-500/20 border border-sky-500/30 flex items-center justify-center">
                <span className="text-[10px] font-bold text-sky-300">KW</span>
              </div>
              <div className="flex items-center gap-1.5 text-xs text-white/70">
                <span>Thinking</span>
                <TypingDots />
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Input area */}
      <div className="relative z-10 shrink-0 border-t border-border bg-card/60 backdrop-blur-sm px-4 py-3">
        <div className="max-w-3xl mx-auto">
          {/* Input card */}
          <motion.div
            className="relative backdrop-blur-xl bg-white/[0.02] rounded-2xl border border-white/[0.06] shadow-xl"
            initial={{ scale: 0.99, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{ duration: 0.3 }}
          >
            {/* Textarea */}
            <div className="px-4 pt-3 pb-1">
              <AnimatedTextarea
                ref={textareaRef}
                value={input}
                onChange={(e) => {
                  setInput(e.target.value);
                  adjustHeight();
                }}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    send();
                  }
                }}
                onFocus={() => setInputFocused(true)}
                onBlur={() => setInputFocused(false)}
                placeholder={
                  isCurrentLoading
                    ? "Type to queue next question…"
                    : "Ask about the codebase…"
                }
                containerClassName="w-full"
                className={cn(
                  "w-full resize-none bg-transparent border-none text-sm",
                  "text-foreground focus:outline-none",
                  "placeholder:text-muted-foreground/50 min-h-[44px]"
                )}
                style={{ overflow: "hidden" }}
                showRing={false}
              />
            </div>

            {/* Toolbar */}
            <div className="px-3 pb-3 flex items-center justify-between gap-2">
              <div className="flex items-center gap-1">
                <motion.button
                  type="button"
                  whileTap={{ scale: 0.92 }}
                  className="p-2 text-muted-foreground hover:text-foreground rounded-lg transition-colors hover:bg-accent/50"
                  title="Attach file"
                >
                  <Paperclip className="w-4 h-4" />
                </motion.button>
                <motion.button
                  type="button"
                  whileTap={{ scale: 0.92 }}
                  className="p-2 text-muted-foreground hover:text-foreground rounded-lg transition-colors hover:bg-accent/50"
                  title="Commands"
                >
                  <Command className="w-4 h-4" />
                </motion.button>
              </div>

              <div className="flex items-center gap-2">
                {isCurrentLoading && currentQueueCount > 0 && (
                  <motion.span
                    className="text-xs text-sky-400/80"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                  >
                    {currentQueueCount} queued
                  </motion.span>
                )}
                <motion.button
                  type="button"
                  onClick={send}
                  disabled={!input.trim()}
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.96 }}
                  className={cn(
                    "flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-sm font-medium transition-all",
                    input.trim()
                      ? isCurrentLoading
                        ? "bg-sky-500/20 text-sky-300 border border-sky-500/30"
                        : "bg-primary text-primary-foreground shadow-lg"
                      : "bg-white/[0.04] text-muted-foreground"
                  )}
                >
                  {isCurrentLoading ? (
                    <>
                      <LoaderIcon className="w-3.5 h-3.5 animate-spin" />
                      <span>Queue</span>
                    </>
                  ) : (
                    <>
                      <ArrowUpIcon className="w-3.5 h-3.5" />
                      <span>Send</span>
                    </>
                  )}
                </motion.button>
              </div>
            </div>
          </motion.div>

          <p className="text-center text-xs text-muted-foreground/40 mt-1.5">
            {isCurrentLoading && currentQueueCount > 0
              ? `${currentQueueCount} question${currentQueueCount > 1 ? "s" : ""} queued — runs automatically`
              : "Shift+Enter for new line"}
          </p>
        </div>
      </div>

      {/* Mouse-follow gradient when input is focused */}
      {inputFocused && (
        <motion.div
          className="fixed w-[40rem] h-[40rem] rounded-full pointer-events-none z-0 opacity-[0.025] bg-gradient-to-r from-sky-500 via-cyan-400 to-blue-500 blur-[96px]"
          animate={{ x: mousePosition.x - 320, y: mousePosition.y - 320 }}
          transition={{ type: "spring", damping: 25, stiffness: 150, mass: 0.5 }}
        />
      )}
    </div>
  );
}
