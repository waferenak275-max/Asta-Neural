import { useState, useEffect, useRef, useCallback } from "react";

/* ─── CSS ────────────────────────────────────────────────────────────────── */
const CSS = `
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

/* ── Light mode ── */
:root {
  --bg:        #f7f6f4;
  --surface:   #ffffff;
  --surface2:  #f0eeeb;
  --border:    #e5e2dd;
  --text:      #1a1816;
  --muted:     #8c8680;
  --accent:    #d4a96a;
  --asta:      #3d3530;
  --tag-bg:    #edeae5;
  --r:         14px;
  --rs:        8px;
  --shadow:    0 2px 16px rgba(0,0,0,0.07);
  --font:      'Sora', sans-serif;
  --mono:      'JetBrains Mono', monospace;
  --ease:      0.22s cubic-bezier(0.4,0,0.2,1);
}

/* ── Dark mode — applied via class on <html> ── */
html.dark {
  --bg:        #141210;
  --surface:   #1e1b18;
  --surface2:  #252018;
  --border:    #2e2a24;
  --text:      #e8e0d5;
  --muted:     #6b6560;
  --accent:    #d4a96a;
  --asta:      #c8a882;
  --tag-bg:    #2a2520;
  --shadow:    0 2px 16px rgba(0,0,0,0.35);
}

html, body { height: 100%; width: 100%; overflow: hidden; font-family: var(--font); background: var(--bg); color: var(--text); transition: background 0.3s ease, color 0.3s ease; }
#root { height: 100%; width: 100%; display: flex; flex-direction: column; }

::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 99px; }

@keyframes fadeIn  { from { opacity:0 } to { opacity:1 } }
@keyframes slideR  { from { opacity:0; transform:translateX(20px) } to { opacity:1; transform:none } }
@keyframes slideL  { from { opacity:0; transform:translateX(-20px) } to { opacity:1; transform:none } }
@keyframes slideP  { from { opacity:0; transform:translateX(24px) } to { opacity:1; transform:none } }
@keyframes pulse   { 0%,100% { opacity:1 } 50% { opacity:.3 } }
@keyframes blink   { 0%,100% { opacity:1 } 50% { opacity:0 } }
@keyframes spin    { to { transform:rotate(360deg) } }
@keyframes waveIn  { from { opacity:0; transform:scale(.94) } to { opacity:1; transform:none } }

@keyframes tokenFade { from { opacity:0 } to { opacity:1 } }
.stream-token { animation: tokenFade 0.38s ease forwards; }

/* Dark mode toggle pill */
.dm-toggle {
  position: relative;
  width: 40px; height: 22px;
  background: var(--border);
  border-radius: 99px;
  cursor: pointer;
  border: none;
  transition: background 0.25s ease;
  flex-shrink: 0;
}
.dm-toggle.on { background: var(--accent); }
.dm-toggle::after {
  content: '';
  position: absolute;
  top: 3px; left: 3px;
  width: 16px; height: 16px;
  border-radius: 50%;
  background: white;
  transition: transform 0.25s cubic-bezier(0.4,0,0.2,1);
}
.dm-toggle.on::after { transform: translateX(18px); }
`;

/* ─── Constants ──────────────────────────────────────────────────────────── */
const WS_URL  = "ws://localhost:8000/ws/chat";
const API_URL = "http://localhost:8000";

const EMOTION_MAP = {
  netral:   { emoji: "*",  color: "#8c8680", label: "Netral"   },
  senang:   { emoji: "✦", color: "#d4a96a", label: "Senang"   },
  romantis: { emoji: "♡", color: "#c97b8a", label: "Romantis" },
  sedih:    { emoji: "·",  color: "#7a9ec7", label: "Sedih"    },
  cemas:    { emoji: "~",  color: "#b07ab0", label: "Cemas"    },
  marah:    { emoji: "!",  color: "#c07060", label: "Marah"    },
};

/* ─── Inject CSS once ────────────────────────────────────────────────────── */
if (!document.getElementById("asta-css")) {
  const tag = document.createElement("style");
  tag.id = "asta-css";
  tag.textContent = CSS;
  document.head.appendChild(tag);
}

/* ══════════════════ MAIN COMPONENT ══════════════════════════════════════════ */
export default function AstaUI() {
  const [messages,    setMessages]    = useState([]);
  const [input,       setInput]       = useState("");
  const [connected,   setConnected]   = useState(false);
  const [thinking,    setThinking]    = useState(false);
  const [streaming,   setStreaming]   = useState(false);
  const [emotion,     setEmotion]     = useState({ user_emotion: "netral", intensity: "rendah", trend: "stabil" });
  const [thought,     setThought]     = useState(null);
  const [memory,      setMemory]      = useState(null);
  const [showMemory,  setShowMemory]  = useState(false);
  const [showThought, setShowThought] = useState(false);
  const [serverReady, setServerReady] = useState(false);
  const [thoughtEnabled, setThoughtEnabled] = useState(true);
  const [darkMode,    setDarkMode]    = useState(() => {
    // Persist preference
    return localStorage.getItem("asta-dark") === "1";
  });

  const wsRef      = useRef(null);
  const bottomRef  = useRef(null);
  const bufRef     = useRef("");
  const msgIdRef   = useRef(0);
  const tokenIdRef = useRef(0);

  /* Apply / remove dark class on <html> */
  useEffect(() => {
    document.documentElement.classList.toggle("dark", darkMode);
    localStorage.setItem("asta-dark", darkMode ? "1" : "0");
  }, [darkMode]);

  const sanitize = (text) => {
    if (!text) return "";
    return text.replace(/\uFFFD/g, "").replace(/[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]/g, "");
  };

  const scrollBottom = useCallback(() => {
    setTimeout(() => bottomRef.current?.scrollIntoView({ behavior: "smooth" }), 60);
  }, []);

  /* Poll server */
  useEffect(() => {
    let iv = setInterval(async () => {
      try {
        const r = await fetch(`${API_URL}/status`);
        const d = await r.json();
        if (d.ready) { setServerReady(true); clearInterval(iv); }
      } catch (_) {}
    }, 2000);
    return () => clearInterval(iv);
  }, []);

  /* WebSocket */
  useEffect(() => {
    if (!serverReady) return;
    const connect = () => {
      const ws = new WebSocket(WS_URL);
      wsRef.current = ws;
      ws.onopen  = () => setConnected(true);
      ws.onclose = () => { setConnected(false); setTimeout(connect, 2000); };
      ws.onerror = () => ws.close();
      ws.onmessage = (e) => {
        const msg = JSON.parse(e.data);
        if (msg.type === "thinking_start") {
          setThinking(true);
        } else if (msg.type === "thought") {
          setThinking(false);
          setThought(msg.data);
          if (msg.data.emotion) setEmotion(msg.data.emotion);
        } else if (msg.type === "stream_start") {
          bufRef.current = "";
          tokenIdRef.current = 0;
          setStreaming(true);
          const id = ++msgIdRef.current;
          setMessages(p => [...p, { id, role: "assistant", content: "", tokens: [] }]);
        } else if (msg.type === "chunk") {
          const clean = sanitize(msg.text);
          if (!clean) return;
          bufRef.current += clean;
          const buf = bufRef.current;
          const tid = ++tokenIdRef.current;
          setMessages(p => p.map(m =>
            m.id === msgIdRef.current
              ? { ...m, content: buf, tokens: [...(m.tokens || []), { id: tid, text: clean }] }
              : m
          ));
          scrollBottom();
        } else if (msg.type === "stream_end") {
          setStreaming(false);
          setMessages(p => p.map(m =>
            m.id === msgIdRef.current ? { ...m, tokens: null } : m
          ));
          scrollBottom();
          fetchMemory();
        }
      };
    };
    connect();
    return () => wsRef.current?.close();
  }, [serverReady]);

  const fetchMemory = useCallback(async () => {
    try { setMemory(await (await fetch(`${API_URL}/memory`)).json()); } catch (_) {}
  }, []);

  const fetchConfig = useCallback(async () => {
    try {
      const d = await (await fetch(`${API_URL}/config`)).json();
      setThoughtEnabled(d.internal_thought_enabled ?? true);
    } catch (_) {}
  }, []);

  useEffect(() => { if (serverReady) { fetchMemory(); fetchConfig(); } }, [serverReady]);

  const send = useCallback(() => {
    const text = input.trim();
    if (!text || !connected || thinking || streaming) return;
    setMessages(p => [...p, { id: ++msgIdRef.current, role: "user", content: text }]);
    setInput("");
    scrollBottom();
    wsRef.current?.send(JSON.stringify({ message: text }));
  }, [input, connected, thinking, streaming, scrollBottom]);

  const saveSession = async () => {
    await fetch(`${API_URL}/save`, { method: "POST" });
    fetchMemory();
  };

  const toggleThought = async () => {
    try {
      const d = await (await fetch(`${API_URL}/config/thought`, { method: "POST" })).json();
      setThoughtEnabled(d.internal_thought_enabled);
    } catch (_) {}
  };

  const emo = EMOTION_MAP[emotion.user_emotion] || EMOTION_MAP.netral;

  const statusText = !serverReady ? "Memuat model…"
    : !connected ? "Menghubungkan…"
    : thinking   ? "Berpikir…"
    : streaming  ? "Mengetik…"
    : "Online";

  return (
    <div style={S.root}>

      {/* Top bar */}
      <div style={S.topBar}>
        <ToggleBtn active={showThought} onClick={() => setShowThought(p => !p)} icon="⟡" label="Thought" />
        <ToggleBtn active={showMemory}  onClick={() => setShowMemory(p => !p)}  icon="◈" label="Memory"  />

        {/* Spacer */}
        <div style={{ flex: 1 }} />

        {/* Thought enabled/disabled toggle */}
        <div style={{ display: "flex", alignItems: "center", gap: 7 }}>
          <span style={{ fontSize: 11, color: thoughtEnabled ? "var(--accent)" : "var(--muted)", fontFamily: "var(--mono)", userSelect: "none", transition: "color 0.2s" }}>
            {thoughtEnabled ? "⟡ on" : "⟡ off"}
          </span>
          <button
            className={`dm-toggle${thoughtEnabled ? " on" : ""}`}
            onClick={toggleThought}
            title={thoughtEnabled ? "Matikan Internal Thought" : "Hidupkan Internal Thought"}
          />
        </div>

        <div style={{ width: 1, height: 18, background: "var(--border)" }} />

        {/* Dark mode toggle */}
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span style={{ fontSize: 12, color: "var(--muted)", fontFamily: "var(--mono)", userSelect: "none" }}>
            {darkMode ? "☾" : "○"}
          </span>
          <button
            className={`dm-toggle${darkMode ? " on" : ""}`}
            onClick={() => setDarkMode(p => !p)}
            title={darkMode ? "Mode Terang" : "Mode Gelap"}
          />
        </div>
      </div>

      {/* Layout */}
      <div style={S.layout}>

        {/* Thought panel */}
        <SidePanel visible={showThought} side="left" title="Internal Thought" icon="⟡">
          <ThoughtPanel thought={thought} thinking={thinking} />
        </SidePanel>

        {/* Chat */}
        <div style={S.chatCol}>
          {/* Header */}
          <div style={S.header}>
            <div style={S.hLeft}>
              <Avatar emo={emo} />
              <div>
                <div style={S.hName}>Asta</div>
                <div style={S.hSub}>{statusText}</div>
              </div>
            </div>
            <div style={S.hRight}>
              <EmoBadge emotion={emotion} emo={emo} />
              <button onClick={saveSession} style={S.saveBtn}>↓ Simpan</button>
            </div>
          </div>

          {/* Messages */}
          <div style={S.msgList}>
            {messages.length === 0 && (
              <div style={S.empty}>
                <div style={{ fontSize: 42, marginBottom: 14, animation: "pulse 3s ease infinite" }}>{emo.emoji}</div>
                <div style={{ fontSize: 18, fontWeight: 500 }}>Halo, Aditiya~</div>
                <div style={{ fontSize: 14, color: "var(--muted)", marginTop: 5 }}>Asta siap ngobrol denganmu.</div>
              </div>
            )}
            {messages.map(m => (
              <Bubble key={m.id} msg={m} isStreaming={streaming && m.id === msgIdRef.current} />
            ))}
            {thinking && <ThinkingBubble />}
            <div ref={bottomRef} />
          </div>

          {/* Input */}
          <div style={S.inputWrap}>
            <div style={S.inputRow}>
              <textarea
                value={input}
                onChange={e => setInput(e.target.value)}
                onKeyDown={e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); } }}
                placeholder="Tulis pesan…"
                style={S.textarea}
                rows={1}
                disabled={!connected || thinking || streaming}
              />
              <button
                onClick={send}
                disabled={!connected || thinking || streaming || !input.trim()}
                style={{ ...S.sendBtn, opacity: (!connected || thinking || streaming || !input.trim()) ? 0.4 : 1 }}
              >↑</button>
            </div>
            <div style={S.hint}>Enter kirim · Shift+Enter baris baru</div>
          </div>
        </div>

        {/* Memory panel */}
        <SidePanel visible={showMemory} side="right" title="Memory" icon="◈">
          <MemoryPanel memory={memory} onRefresh={fetchMemory} />
        </SidePanel>
      </div>
    </div>
  );
}

/* ══════════════════ SUB-COMPONENTS ══════════════════════════════════════════ */

function Avatar({ emo }) {
  return (
    <div style={{ position: "relative", marginRight: 10 }}>
      <div style={{
        width: 45, height: 45, borderRadius: "50%",
        background: `linear-gradient(135deg, ${emo.color}33, ${emo.color}11)`,
        border: `1.5px solid ${emo.color}55`,
        display: "flex", alignItems: "center", justifyContent: "center",
        fontSize: 22, lineHeight: 1, transition: "all 0.4s ease",
      }}>{emo.emoji}</div>
      <div style={{
        position: "absolute", bottom: 1, right: 1,
        width: 10, height: 10, borderRadius: "50%",
        background: "#2e7d57", border: "2px solid var(--bg)",
      }} />
    </div>
  );
}

function EmoBadge({ emotion, emo }) {
  return (
    <div style={{
      display: "flex", alignItems: "center", gap: 7,
      padding: "4px 12px", borderRadius: 99,
      background: `${emo.color}14`, border: `1px solid ${emo.color}30`,
      fontSize: 12, fontWeight: 500, color: emo.color,
      fontFamily: "var(--mono)", transition: "all 0.4s ease",
    }}>
      {/* Emoji di badge juga diperbesar */}
      <span style={{ fontSize: 16 }}>{emo.emoji}</span>
      {emo.label}
      <span style={{ color: "var(--muted)", fontWeight: 400 }}>· {emotion.intensity}</span>
    </div>
  );
}

function Bubble({ msg, isStreaming }) {
  const isUser = msg.role === "user";
  return (
    <div style={{
      display: "flex",
      justifyContent: isUser ? "flex-end" : "flex-start",
      padding: "3px 0",
      animation: `${isUser ? "slideR" : "slideL"} 0.28s ease`,
    }}>
      <div style={{
        maxWidth: "68%",
        padding: "11px 16px",
        borderRadius: isUser ? "16px 4px 16px 16px" : "4px 16px 16px 16px",
        background: isUser ? "var(--asta)" : "var(--surface)",
        color: isUser ? "#f5f0eb" : "var(--text)",
        fontSize: 13, lineHeight: 1.65,
        boxShadow: "var(--shadow)",
        border: isUser ? "none" : "1px solid var(--border)",
        wordBreak: "break-word", whiteSpace: "pre-wrap",
        textAlign: "left",
      }}>
        {isStreaming && msg.tokens
          ? <>
              {msg.tokens.map(t => (
                <span key={t.id} className="stream-token">{t.text}</span>
              ))}
              <span style={{
                display: "inline-block", width: 7, height: 14,
                background: "var(--accent)", borderRadius: 2, marginLeft: 2,
                verticalAlign: "text-bottom", animation: "blink 0.8s step-end infinite",
              }} />
            </>
          : <>
              {msg.content}
              {isStreaming && (
                <span style={{
                  display: "inline-block", width: 7, height: 14,
                  background: "var(--accent)", borderRadius: 2, marginLeft: 2,
                  verticalAlign: "text-bottom", animation: "blink 0.8s step-end infinite",
                }} />
              )}
            </>
        }
      </div>
    </div>
  );
}

function ThinkingBubble() {
  return (
    <div style={{ display: "flex", padding: "3px 0", animation: "fadeIn 0.3s ease" }}>
      <div style={{
        padding: "13px 18px", borderRadius: "4px 16px 16px 16px",
        background: "var(--surface)", border: "1px solid var(--border)",
        boxShadow: "var(--shadow)", display: "flex", gap: 6, alignItems: "center",
      }}>
        {[0, 0.18, 0.36].map(d => (
          <div key={d} style={{
            width: 6, height: 6, borderRadius: "50%", background: "var(--accent)",
            animation: `pulse 1.2s ease-in-out ${d}s infinite`,
          }} />
        ))}
      </div>
    </div>
  );
}

function ToggleBtn({ active, onClick, icon, label }) {
  return (
    <button onClick={onClick} style={{
      display: "flex", alignItems: "center", gap: 6,
      padding: "6px 14px", borderRadius: 99,
      background: active ? "var(--asta)" : "var(--surface)",
      color: active ? (document.documentElement.classList.contains("dark") ? "#1a1816" : "#f5f0eb") : "var(--muted)",
      border: `1px solid ${active ? "var(--asta)" : "var(--border)"}`,
      fontSize: 12, fontFamily: "var(--font)", fontWeight: 500,
      cursor: "pointer", transition: "all var(--ease)",
    }}>
      {icon} {label}
    </button>
  );
}

function SidePanel({ visible, side, title, icon, children }) {
  return (
    <div style={{
      width: visible ? 250 : 0, minWidth: visible ? 250 : 0,
      overflow: "hidden", flexShrink: 0,
      transition: "width 0.3s cubic-bezier(0.4,0,0.2,1), min-width 0.3s cubic-bezier(0.4,0,0.2,1)",
    }}>
      <div style={{
        width: 250, height: "100%",
        background: "var(--surface)",
        borderLeft:  side === "right" ? "1px solid var(--border)" : "none",
        borderRight: side === "left"  ? "1px solid var(--border)" : "none",
        display: "flex", flexDirection: "column",
        opacity: visible ? 1 : 0,
        transition: "opacity 0.3s ease",
      }}>
        <div style={{
          padding: "14px 16px 10px",
          borderBottom: "1px solid var(--border)",
          display: "flex", alignItems: "center", gap: 8,
          fontSize: 11, fontWeight: 600, letterSpacing: "0.06em",
          color: "var(--muted)", textTransform: "uppercase", flexShrink: 0,
        }}>
          {icon} {title}
        </div>
        <div style={{ flex: 1, overflowY: "auto", padding: 14 }}>
          {children}
        </div>
      </div>
    </div>
  );
}

function ThoughtPanel({ thought, thinking }) {
  if (thinking) return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", paddingTop: 40 }}>
      <div style={{
        width: 18, height: 18, borderRadius: "50%",
        border: "2px solid var(--border)", borderTop: "2px solid var(--accent)",
        animation: "spin 0.8s linear infinite", marginBottom: 10,
      }} />
      <span style={{ fontSize: 11, color: "var(--muted)" }}>Berpikir…</span>
    </div>
  );

  if (!thought) return (
    <div style={{ color: "var(--muted)", fontSize: 12, textAlign: "center", paddingTop: 40 }}>
      Belum ada thought
    </div>
  );

  const rows = [
    { label: "Need Search", value: thought.need_search ? "✓ Ya" : "✗ Tidak", mono: true },
    thought.search_query && { label: "Query",  value: thought.search_query, mono: true },
    thought.recall_topic && { label: "Recall", value: thought.recall_topic },
    { label: "Tone", value: thought.tone },
    thought.note && { label: "Note", value: thought.note },
  ].filter(Boolean);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 6, animation: "fadeIn 0.3s ease" }}>
      {rows.map((row, i) => (
        /* Card dikurangi 25%: padding 10→7, gap 10→6, font 13→11 */
        <div key={i} style={S.card}>
          <div style={S.cardLabel}>{row.label}</div>
          <div style={{ fontSize: 11, color: "var(--text)", lineHeight: 1.45, fontFamily: row.mono ? "var(--mono)" : "var(--font)" }}>
            {row.value}
          </div>
        </div>
      ))}
      {thought.emotion && (
        <div style={{ marginTop: 4 }}>
          <div style={S.sectionTitle}>Emotion State</div>
          <EmotionDetail emotion={thought.emotion} />
        </div>
      )}
    </div>
  );
}

function EmotionDetail({ emotion }) {
  const emo = EMOTION_MAP[emotion.user_emotion] || EMOTION_MAP.netral;
  return (
    <div style={{
      background: `${emo.color}0e`,
      border: `1px solid ${emo.color}25`,
      borderRadius: "var(--rs)",
      /* Padding dikurangi 25% */
      padding: "7px 9px", marginTop: 5,
      animation: "waveIn 0.35s ease",
    }}>
      {[["Emosi", `${emo.emoji} ${emo.label}`], ["Intensitas", emotion.intensity], ["Tren", emotion.trend]].map(([k, v]) => (
        <div key={k} style={{ display: "flex", justifyContent: "space-between", marginBottom: 0, marginTop: 2, fontSize: 11 }}>
          <span style={{ color: "var(--muted)" }}>{k}</span>
          <span style={{ color: emo.color, fontWeight: 500, fontFamily: "var(--mono)" }}>{v}</span>
        </div>
      ))}
    </div>
  );
}

function MemoryPanel({ memory, onRefresh }) {
  if (!memory) return (
    <div style={{ color: "var(--muted)", fontSize: 13, textAlign: "center", paddingTop: 40 }}>
      Memuat memory…
    </div>
  );

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 18, animation: "fadeIn 0.3s ease" }}>

      {memory.profile?.preferensi?.length > 0 && (
        <div>
          <div style={S.sectionTitle}>Preferensi</div>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 5, marginTop: 7 }}>
            {memory.profile.preferensi.map((p, i) => (
              <span key={i} style={S.tag}>{p}</span>
            ))}
          </div>
        </div>
      )}

      {memory.recent_facts && (
        <div>
          <div style={S.sectionTitle}>Fakta Terbaru</div>
          <div style={{ fontFamily: "var(--mono)", fontSize: 12, color: "var(--muted)", lineHeight: 1.8, marginTop: 7, whiteSpace: "pre-wrap" }}>
            {memory.recent_facts}
          </div>
        </div>
      )}

      {memory.core && (
        <div>
          <div style={S.sectionTitle}>Core Summary</div>
          <div style={{ fontSize: 13, lineHeight: 1.7, marginTop: 7, padding: "10px 12px", background: "var(--surface2)", borderRadius: "var(--rs)" }}>
            {memory.core}
          </div>
        </div>
      )}

      {memory.sessions?.length > 0 && (
        <div>
          <div style={S.sectionTitle}>Sesi Tersimpan</div>
          <div style={{ display: "flex", flexDirection: "column", gap: 7, marginTop: 7 }}>
            {memory.sessions.map((s, i) => (
              <div key={i} style={S.card}>
                <div style={{ fontSize: 11, color: "var(--muted)", fontFamily: "var(--mono)" }}>
                  {new Date(s.timestamp).toLocaleString("id-ID", { dateStyle: "short", timeStyle: "short" })} · {s.facts} fakta
                </div>
                <div style={{ fontSize: 12, marginTop: 3, lineHeight: 1.5 }}>{s.preview}</div>
              </div>
            ))}
          </div>
        </div>
      )}

      <button onClick={onRefresh} style={S.refreshBtn}>↻ Refresh</button>
    </div>
  );
}

/* ─── Styles ─────────────────────────────────────────────────────────────── */
const S = {
  root: {
    display: "flex", flexDirection: "column",
    width: "100%", height: "100vh", overflow: "hidden",
    background: "var(--bg)",
  },
  topBar: {
    display: "flex", alignItems: "center", gap: 8,
    padding: "10px 20px", flexShrink: 0,
    borderBottom: "1px solid var(--border)", background: "var(--bg)",
  },
  layout: {
    flex: 1, display: "flex", overflow: "hidden", minHeight: 0,
  },
  chatCol: {
    flex: 1, display: "flex", flexDirection: "column", minWidth: 0,
  },
  header: {
    display: "flex", alignItems: "center", justifyContent: "space-between",
    padding: "14px 24px", flexShrink: 0,
    borderBottom: "1px solid var(--border)", background: "var(--surface)",
  },
  hLeft:  { display: "flex", alignItems: "center" },
  hName:  { fontSize: 17, textAlign: "left", fontWeight: 600, letterSpacing: "-0.01em", lineHeight: 1.4 },
  hSub:   { fontSize: 12, textAlign: "left", color: "var(--muted)", marginTop: 1, fontFamily: "var(--mono)", lineHeight: 1.3 },
  hRight: { display: "flex", alignItems: "center", gap: 12 },
  msgList: {
    flex: 1, overflowY: "auto", padding: "24px 32px",
    display: "flex", flexDirection: "column", gap: 4, minHeight: 0,
  },
  empty: {
    display: "flex", flex: 1, flexDirection: "column", alignItems: "center",
    justifyContent: "center", opacity: 0.55,
    animation: "fadeIn 0.6s ease",
  },
  inputWrap: {
    padding: "14px 24px 18px", flexShrink: 0,
    borderTop: "1px solid var(--border)", background: "var(--surface)",
  },
  inputRow: { display: "flex", gap: 12, alignItems: "flex-end" },
  textarea: {
    flex: 1, resize: "none", padding: "12px 16px",
    borderRadius: "var(--r)", border: "1.5px solid var(--border)",
    background: "var(--surface2)", fontSize: 15, fontFamily: "var(--font)",
    color: "var(--text)", lineHeight: 1.6, outline: "none",
    maxHeight: 140, overflowY: "auto",
  },
  sendBtn: {
    width: 50, height: 50, borderRadius: "50%",
    background: "var(--asta)", color: "#f5f0eb",
    border: "none", fontSize: 18, cursor: "pointer",
    transition: "all var(--ease)", flexShrink: 0,
    display: "flex", alignItems: "center", justifyContent: "center", fontWeight: 600,
  },
  hint: { fontSize: 11, color: "var(--muted)", marginTop: 7, fontFamily: "var(--mono)" },
  saveBtn: {
    padding: "6px 14px", borderRadius: 99,
    background: "transparent", border: "1px solid var(--border)",
    color: "var(--muted)", fontSize: 12, fontFamily: "var(--font)", cursor: "pointer",
  },
  /* Thought card dikurangi 25%: padding 10/12 → 7/9 */
  card: {
    padding: "7px 9px", borderRadius: "var(--rs)",
    background: "var(--surface2)", border: "1px solid var(--border)",
  },
  cardLabel: {
    fontSize: 9, color: "var(--muted)", textTransform: "uppercase",
    letterSpacing: "0.06em", marginBottom: 0, fontWeight: 600,
  },
  sectionTitle: {
    fontSize: 11, fontWeight: 600, color: "var(--muted)",
    textTransform: "uppercase", marginBottom: 3, letterSpacing: "0.06em",
  },
  tag: {
    padding: "4px 11px", borderRadius: 99,
    background: "var(--tag-bg)", fontSize: 12, color: "var(--text)", fontWeight: 500,
  },
  refreshBtn: {
    width: "100%", padding: "9px", borderRadius: "var(--rs)",
    border: "1px solid var(--border)", background: "transparent",
    cursor: "pointer", fontSize: 13, color: "var(--muted)", fontFamily: "var(--font)",
  },
};