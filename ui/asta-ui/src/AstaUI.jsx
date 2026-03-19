import { useState, useEffect, useRef, useCallback } from "react";

const CSS = `
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
:root {
  --bg:#f7f6f4; --surface:#fff; --surface2:#f0eeeb; --border:#e5e2dd;
  --text:#1a1816; --muted:#8c8680; --accent:#d4a96a; --asta:#3d3530;
  --tag-bg:#edeae5; --r:14px; --rs:8px;
  --shadow:0 2px 16px rgba(0,0,0,0.07);
  --font:'Sora',sans-serif; --mono:'JetBrains Mono',monospace;
  --ease:0.22s cubic-bezier(0.4,0,0.2,1);
  --blue:#7a9ec7; --green:#6ab87a; --rose:#c97b8a; --purple:#9b7ac9;
}
html.dark {
  --bg:#141210; --surface:#1e1b18; --surface2:#252018; --border:#2e2a24;
  --text:#e8e0d5; --muted:#6b6560; --accent:#d4a96a; --asta:#c8a882;
  --tag-bg:#2a2520; --shadow:0 2px 16px rgba(0,0,0,0.35);
}
html,body{height:100%;width:100%;overflow:hidden;font-family:var(--font);background:var(--bg);color:var(--text);transition:background .3s,color .3s;}
#root{height:100%;width:100%;display:flex;flex-direction:column;}
::-webkit-scrollbar{width:5px} ::-webkit-scrollbar-thumb{background:var(--border);border-radius:99px}
@keyframes fadeIn{from{opacity:0}to{opacity:1}}
@keyframes slideR{from{opacity:0;transform:translateX(20px)}to{opacity:1;transform:none}}
@keyframes slideL{from{opacity:0;transform:translateX(-20px)}to{opacity:1;transform:none}}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}
@keyframes blink{0%,100%{opacity:1}50%{opacity:0}}
@keyframes spin{to{transform:rotate(360deg)}}
@keyframes waveIn{from{opacity:0;transform:scale(.94)}to{opacity:1;transform:none}}
@keyframes pullOut{from{opacity:0;transform:translateY(8px) scale(.98)}to{opacity:1;transform:translateY(0) scale(1)}}
@keyframes popIn{from{opacity:0;transform:scale(.9) translateX(-10px)}to{opacity:1;transform:scale(1) translateX(0)}}
@keyframes tokenFade{from{opacity:0}to{opacity:1}}
.stream-token{animation:tokenFade .38s ease forwards}
.note-bubble{animation:popIn .3s cubic-bezier(0.4,0,0.2,1) forwards; transform-origin: left center;}
.note-bubble::after{content:'';position:absolute;left:-7px;top:14px;width:12px;height:12px;background:var(--surface);border-left:1.5px solid var(--green);border-bottom:1.5px solid var(--green);transform:rotate(45deg);z-index:1}
.dm-toggle{position:relative;width:40px;height:22px;background:var(--border);border-radius:99px;cursor:pointer;border:none;transition:background .25s;flex-shrink:0}

.dm-toggle.on{background:var(--accent)}
.dm-toggle::after{content:'';position:absolute;top:3px;left:3px;width:16px;height:16px;border-radius:50%;background:white;transition:transform .25s cubic-bezier(0.4,0,0.2,1)}
.dm-toggle.on::after{transform:translateX(18px)}
.bar-fill{transition:width .6s ease}
`;

const WS_URL  = "ws://localhost:8000/ws/chat";
const API_URL = "http://localhost:8000";

const EMO_MAP = {
  netral:       { emoji:"*",  color:"#8c8680", label:"Netral"    },
  senang:       { emoji:"✦", color:"#d4a96a", label:"Senang"    },
  romantis:     { emoji:"♡", color:"#c97b8a", label:"Romantis"  },
  sedih:        { emoji:"·",  color:"#7a9ec7", label:"Sedih"     },
  cemas:        { emoji:"~",  color:"#b07ab0", label:"Cemas"     },
  marah:        { emoji:"!",  color:"#c07060", label:"Marah"     },
  rindu:        { emoji:"◦",  color:"#a07ab0", label:"Rindu"     },
  bangga:       { emoji:"★", color:"#d4a96a", label:"Bangga"    },
  kecewa:       { emoji:"…",  color:"#9ab0c7", label:"Kecewa"    },
  "sangat senang": { emoji:"✦✦", color:"#d4a96a", label:"Sangat Senang" },
  "sedikit senang":{ emoji:"·",  color:"#c8b87a", label:"Sedikit Senang" },
  murung:       { emoji:"·",  color:"#7a9ec7", label:"Murung"    },
  "sangat murung": { emoji:"··", color:"#6080a7", label:"Sangat Murung" },
};

const getEmo = (key) => EMO_MAP[key] || EMO_MAP.netral;

if (!document.getElementById("asta-css")) {
  const t = document.createElement("style");
  t.id = "asta-css"; t.textContent = CSS;
  document.head.appendChild(t);
}

export default function AstaUI() {
  const [messages,    setMessages]    = useState([]);
  const [input,       setInput]       = useState("");
  const [connected,   setConnected]   = useState(false);
  const [thinking,    setThinking]    = useState(false);
  const [streaming,   setStreaming]   = useState(false);
  const [userEmotion, setUserEmotion] = useState({ user_emotion:"netral", intensity:"rendah", trend:"stabil" });
  const [astaEmotion, setAstaEmotion] = useState({ current_emotion:"netral", mood:"netral", mood_score:0, affection_level:0.7, energy_level:0.8 });
  const [thought,     setThought]     = useState(null);
  const [selfModel,   setSelfModel]   = useState(null);
  const [memory,      setMemory]      = useState(null);
  const [panel,       setPanel]       = useState(null); // "thought" | "memory" | "self" | "terminal" | null
  const [statsVisible, setStatsVisible] = useState(false);
  const [sysStats,    setSysStats]    = useState({ cpu: 0, ram: 0, disk: 0 });
  const [noteVisible, setNoteVisible] = useState(false);
  const [serverReady, setServerReady] = useState(false);
  const [thoughtEnabled, setThoughtEnabled] = useState(true);
  const [modelInfo,   setModelInfo]   = useState({ dual_model:false, thought_model:"?", response_model:"?" });
  const [darkMode,    setDarkMode]    = useState(() => localStorage.getItem("asta-dark") === "1");

  const wsRef     = useRef(null);
  const bottomRef = useRef(null);
  const bufRef    = useRef("");
  const msgIdRef  = useRef(0);
  const tokIdRef  = useRef(0);
  const thoughtRef = useRef(null);
  const mainInputRef = useRef(null);

  // Stats message handler from Terminal
  const handleTerminalMessage = useCallback((msg) => {
    if (msg.type === "stats") {
      setSysStats(msg.data);
    }
  }, []);

  // Auto-focus logic
  useEffect(() => {
    if (connected && !thinking && !streaming && panel !== "terminal") {
      mainInputRef.current?.focus();
    }
  }, [connected, thinking, streaming, panel]);

  useEffect(() => {
    document.documentElement.classList.toggle("dark", darkMode);
    localStorage.setItem("asta-dark", darkMode ? "1" : "0");
    // Notify Main Process for TitleBar color sync
    if (window.require) {
      const { ipcRenderer } = window.require('electron');
      ipcRenderer.send('theme-changed', darkMode ? 'dark' : 'light');
    }
  }, [darkMode]);

  const sanitize = t => t ? t.replace(/\uFFFD/g,"").replace(/[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]/g,"") : "";
  const scrollBottom = useCallback(() => { setTimeout(() => bottomRef.current?.scrollIntoView({ behavior:"smooth" }), 60); }, []);

  // Poll server
  useEffect(() => {
    const iv = setInterval(async () => {
      try {
        const d = await (await fetch(`${API_URL}/status`)).json();
        if (d.ready) {
          setServerReady(true);
          setModelInfo({ dual_model: d.dual_model||false, thought_model: d.thought_model||"?", response_model: d.response_model||"?" });
          clearInterval(iv);
        }
      } catch(_) {}
    }, 2000);
    return () => clearInterval(iv);
  }, []);

  // WebSocket
  useEffect(() => {
    if (!serverReady) return;
    const connect = () => {
      const ws = new WebSocket(WS_URL);
      wsRef.current = ws;
      ws.onopen  = () => setConnected(true);
      ws.onclose = () => { setConnected(false); setTimeout(connect, 2000); };
      ws.onerror = () => ws.close();
      ws.onmessage = e => {
        const msg = JSON.parse(e.data);
        if (msg.type === "thinking_start") {
          setThinking(true);
        } else if (msg.type === "thought") {
          setThinking(false);
          setThought(msg.data);
          thoughtRef.current = msg.data;
          if (msg.data.emotion)     setUserEmotion(msg.data.emotion);
          if (msg.data.asta_state)  setAstaEmotion(msg.data.asta_state);
          if (msg.data.model_info)  setModelInfo(msg.data.model_info);
        } else if (msg.type === "stream_start") {
          bufRef.current = ""; tokIdRef.current = 0;
          setStreaming(true);
          const id = ++msgIdRef.current;
          const thoughtData = thoughtRef.current || {};
          const hasWebSearch = Boolean(thoughtData.need_search && thoughtData.search_query);
          setMessages(p => [...p, {
            id,
            role:"assistant",
            content:"",
            tokens:[],
            webSearch: hasWebSearch ? {
              query: thoughtData.search_query || "Web search",
              result: thoughtData.web_result || "Belum ada hasil web search."
            } : null
          }]);
        } else if (msg.type === "chunk") {
          const clean = sanitize(msg.text);
          if (!clean) return;
          bufRef.current += clean;
          const buf = bufRef.current;
          const tid = ++tokIdRef.current;
          setMessages(p => p.map(m =>
            m.id === msgIdRef.current
              ? { ...m, content:buf, tokens:[...(m.tokens||[]), { id:tid, text:clean }] }
              : m
          ));
          scrollBottom();
        } else if (msg.type === "stream_end") {
          setStreaming(false);
          setMessages(p => p.map(m => m.id === msgIdRef.current ? { ...m, tokens:null } : m));
          scrollBottom();
          fetchAll();
        }
      };
    };
    connect();
    return () => wsRef.current?.close();
  }, [serverReady]);

  const fetchAll = useCallback(async () => {
    try { setMemory(await (await fetch(`${API_URL}/memory`)).json()); } catch(_) {}
    try { setSelfModel(await (await fetch(`${API_URL}/self`)).json()); } catch(_) {}
  }, []);

  useEffect(() => {
    if (serverReady) {
      fetchAll();
      fetch(`${API_URL}/config`).then(r=>r.json()).then(d => {
        setThoughtEnabled(d.internal_thought_enabled ?? true);
        setModelInfo({ dual_model: d.dual_model||false, thought_model: d.thought_model||"?", response_model: d.response_model||"?" });
      }).catch(_=>{});
    }
  }, [serverReady]);

  // Auto-show note on thought change
  useEffect(() => {
    if (thought?.note) {
      setNoteVisible(true);
      const timer = setTimeout(() => setNoteVisible(false), 6000);
      return () => clearTimeout(timer);
    }
  }, [thought]);

  const send = useCallback(() => {
    const text = input.trim();
    if (!text || !connected || thinking || streaming) return;
    setMessages(p => [...p, { id:++msgIdRef.current, role:"user", content:text }]);
    setInput("");
    scrollBottom();
    wsRef.current?.send(JSON.stringify({ message: text }));
  }, [input, connected, thinking, streaming, scrollBottom]);

  const toggleThought = async () => {
    try {
      const d = await (await fetch(`${API_URL}/config/thought`, { method:"POST" })).json();
      setThoughtEnabled(d.internal_thought_enabled);
    } catch(_) {}
  };

  const triggerReflect = async () => {
    try {
      await fetch(`${API_URL}/reflect`, { method:"POST" });
      fetchAll();
    } catch(_) {}
  };

  const statusText = !serverReady ? "Memuat model…"
    : !connected  ? "Menghubungkan…"
    : thinking    ? "Berpikir…"
    : streaming   ? "Mengetik…"
    : "Online";

  const emoUser = getEmo(userEmotion.user_emotion);
  const emoAsta = getEmo(astaEmotion.current_emotion || astaEmotion.mood);

  const togglePanel = (name) => setPanel(p => p === name ? null : name);

  return (
    <div style={S.root}>
      {/* Floating Action Plan Bubble (Fixed Position) */}
      {thought?.note && noteVisible && (
        <div className="note-bubble" style={S.noteBubbleFixed}>
          <div style={{fontSize:13,fontWeight:800,color:"var(--green)",marginBottom:4,letterSpacing:"0.05em"}}>Decision Directive</div>
          {thought.note}
        </div>
      )}

      {/* Floating Stats Box */}
      {statsVisible && (
        <div style={S.statsBox}>
          <div style={{fontSize:11,fontWeight:800,color:"var(--accent)",marginBottom:10,letterSpacing:"0.08em",textTransform:"uppercase"}}>System Statistics</div>
          <StatBar label="CPU" value={sysStats.cpu} color="var(--blue)" />
          <StatBar label="RAM" value={sysStats.ram} color="var(--purple)" />
          <StatBar label="DISK" value={sysStats.disk} color="var(--green)" />
        </div>
      )}

      {/* Top bar */}
      <div style={S.topBar}>
        <TopBtn active={panel==="thought"} onClick={()=>togglePanel("thought")} icon="⟡" label="Thought" />
        <TopBtn active={panel==="self"}    onClick={()=>togglePanel("self")}    icon="◉" label="Asta"    />
        <TopBtn active={panel==="memory"}  onClick={()=>togglePanel("memory")}  icon="◈" label="Memory"  />
        {thought?.note && (
          <div style={{marginLeft:0}}>
            <TopBtn 
              active={noteVisible} 
              onClick={() => setNoteVisible(!noteVisible)}
              onMouseEnter={()=>setNoteVisible(true)} 
              onMouseLeave={()=>setNoteVisible(false)}
              icon="#" 
              label="Action" 
            />
          </div>
        )}
        <TopBtn active={panel==="terminal"} onClick={()=>togglePanel("terminal")} icon=">_" label="Terminal" />
        <TopBtn active={statsVisible} onClick={() => setStatsVisible(!statsVisible)} icon="◷" label="Stats" />
        
        <div style={{flex:1}}/>

        {modelInfo.dual_model && (
          <div style={S.modelBadge} title="Thought pakai 3B, Response pakai 8B — KV cache terpisah">
            <span style={{color:"var(--blue)"}}>⟡ {modelInfo.thought_model}</span>
            <span style={{color:"var(--border)"}}>·</span>
            <span style={{color:"var(--accent)"}}>↑ {modelInfo.response_model}</span>
          </div>
        )}

        <div style={{display:"flex",alignItems:"center",gap:7}}>
          <span style={{fontSize:11,color:thoughtEnabled?"var(--accent)":"var(--muted)",fontFamily:"var(--mono)",userSelect:"none"}}>
            {thoughtEnabled ? "⟡ on" : "⟡ off"}
          </span>
          <button className={`dm-toggle${thoughtEnabled?" on":""}`} onClick={toggleThought} title="Toggle Internal Thought" style={{WebkitAppRegion:"no-drag"}}/>
        </div>
        <div style={{width:1,height:18,background:"var(--border)"}}/>
        <div style={{display:"flex",alignItems:"center",gap:8}}>
          <span style={{fontSize:12,color:"var(--muted)",fontFamily:"var(--mono)",userSelect:"none"}}>{darkMode?"☾":"○"}</span>
          <button className={`dm-toggle${darkMode?" on":""}`} onClick={()=>setDarkMode(p=>!p)} title="Dark mode" style={{WebkitAppRegion:"no-drag"}}/>
        </div>
      </div>

      {/* Layout */}
      <div style={S.layout}>

        {/* Thought panel */}
        <SidePanel visible={panel==="thought"} side="left" title="Internal Thought" icon="⟡" width={270}>
          <ThoughtPanel thought={thought} thinking={thinking} modelInfo={modelInfo} />
        </SidePanel>

        {/* Self-model panel */}
        <SidePanel visible={panel==="self"} side="left" title="Asta Self-Model" icon="◉" width={270}>
          <SelfPanel selfModel={selfModel} astaEmotion={astaEmotion} onReflect={triggerReflect} />
        </SidePanel>

        {/* Chat */}
        <div style={S.chatCol}>
          <div style={S.header}>
            <div style={S.hLeft}>
              <Avatar emoAsta={emoAsta} />
              <div>
                <div style={S.hName}>Asta</div>
                <div style={S.hSub}>{statusText}</div>
              </div>
            </div>
            <div style={S.hRight}>
              <AstaEmoBadge asta={astaEmotion} emo={emoAsta} />
              <UserEmoBadge user={userEmotion} emo={emoUser} />
              <button onClick={()=>fetch(`${API_URL}/save`,{method:"POST"}).then(fetchAll)} style={S.saveBtn}>↓</button>
            </div>
          </div>

          <div style={S.msgList}>
            {messages.length === 0 && (
              <div style={S.empty}>
                <div style={{fontSize:44,marginBottom:14,animation:"pulse 3s ease infinite"}}>{emoAsta.emoji}</div>
                <div style={{fontSize:18,fontWeight:500}}>Halo, Aditiya~</div>
                <div style={{fontSize:14,color:"var(--muted)",marginTop:5}}>Asta siap ngobrol denganmu.</div>
              </div>
            )}
            {messages.map(m => <Bubble key={m.id} msg={m} isStreaming={streaming && m.id===msgIdRef.current} />)}
            {thinking && <ThinkingBubble />}
            <div ref={bottomRef}/>
          </div>

          <div style={S.inputWrap}>
            <div style={S.inputRow}>
              <textarea
                ref={mainInputRef}
                value={input}
                onChange={e=>setInput(e.target.value)}
                onKeyDown={e=>{ if(e.key==="Enter"&&!e.shiftKey){e.preventDefault();send();} }}
                placeholder="Tulis pesan…"
                style={S.textarea} rows={1}
                disabled={!connected||thinking||streaming}
              />
              <button onClick={send} disabled={!connected||thinking||streaming||!input.trim()}
                style={{...S.sendBtn,opacity:(!connected||thinking||streaming||!input.trim())?0.4:1}}>↑</button>
            </div>
            <div style={S.hint}>Enter kirim · Shift+Enter baris baru</div>
          </div>
        </div>

        {/* Memory panel */}
        <SidePanel visible={panel==="memory"} side="right" title="Memory" icon="◈" width={260}>
          <MemoryPanel memory={memory} onRefresh={fetchAll} />
        </SidePanel>

        {/* Terminal panel */}
        <SidePanel visible={panel==="terminal"} side="right" title="Terminal" icon=">_" width={450} noPadding={true}>
          <TerminalPanel visible={panel==="terminal"} onMessage={handleTerminalMessage} />
        </SidePanel>
      </div>
    </div>
  );
}

// ── Terminal Component ────────────────────────────────────────────────────────

function TerminalPanel({ visible, onMessage }) {
  const [lines, setLines] = useState(["Asta Terminal Ready.", "Type 'help' for info.", "Commands: start backend, stop backend, cls", ""]);
  const [input, setInput] = useState("");
  const wsRef = useRef(null);
  const scrollRef = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => {
    if (visible) {
      setTimeout(() => inputRef.current?.focus(), 300);
    }
  }, [visible]);

  useEffect(() => {
    // Konek ke port 8001 (Terminal Server Mandiri)
    const ws = new WebSocket("ws://localhost:8001");
    wsRef.current = ws;
    ws.onmessage = (e) => {
      const msg = JSON.parse(e.data);
      if (msg.type === "clear") {
        setLines([]);
      } else if (msg.type === "output") {
        setLines(prev => [...prev, msg.data]);
      } else if (msg.type === "stats") {
        if (onMessage) onMessage(msg);
      }
    };
    ws.onclose = () => setLines(prev => [...prev, "[Disconnected]"]);
    return () => ws.close();
  }, [onMessage]);

  useEffect(() => {
    scrollRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [lines]);

  const runCmd = () => {
    if (!input.trim() || !wsRef.current) return;
    setLines(prev => [...prev, `> ${input}`]);
    wsRef.current.send(input);
    setInput("");
  };

  return (
    <div style={{
      background: "#0c0c0c", color: "#ffffff", fontFamily: "var(--mono)", fontSize: 11,
      height: "100%", display: "flex", flexDirection: "column", padding: "12px 16px", 
      textAlign: "left", lineHeight: 1.4
    }}>
      <div style={{ flex: 1, overflowY: "auto", whiteSpace: "pre-wrap", marginBottom: 10 }}>
        {lines.map((l, i) => <div key={i} style={{ marginBottom: 1 }}>{l}</div>)}
        <div ref={scrollRef} />
      </div>
      <div style={{ display: "flex", gap: 5, borderTop: "1px solid #333", paddingTop: 10 }}>
        <span>$</span>
        <input 
          ref={inputRef}
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={e => e.key === "Enter" && runCmd()}
          style={{
            background: "transparent", border: "none", color: "#ffffff", 
            fontFamily: "var(--mono)", fontSize: 11, outline: "none", flex: 1
          }}
        />
      </div>
    </div>
  );
}

// ── Sub-components ────────────────────────────────────────────────────────────

function Avatar({ emoAsta }) {
  return (
    <div style={{position:"relative",marginRight:10}}>
      <div style={{
        width:45,height:45,borderRadius:"50%",
        background:`linear-gradient(135deg,${emoAsta.color}33,${emoAsta.color}11)`,
        border:`1.5px solid ${emoAsta.color}55`,
        display:"flex",alignItems:"center",justifyContent:"center",
        fontSize:22,lineHeight:1,transition:"all .4s",
      }}>{emoAsta.emoji}</div>
      <div style={{position:"absolute",bottom:1,right:1,width:10,height:10,borderRadius:"50%",background:"#2e7d57",border:"2px solid var(--bg)"}}/>
    </div>
  );
}

function StatBar({ label, value, color }) {
  return (
    <div style={{marginBottom:10}}>
      <div style={{display:"flex",justifyContent:"space-between",fontSize:10,fontFamily:"var(--mono)",color:"var(--muted)",marginBottom:3}}>
        <span>{label}</span>
        <span style={{color}}>{value}%</span>
      </div>
      <div style={{height:6,background:"var(--surface2)",border:"1px solid var(--border)",borderRadius:99,overflow:"hidden"}}>
        <div className="bar-fill" style={{height:"100%",width:`${value}%`,background:color,borderRadius:99}}/>
      </div>
    </div>
  );
}

function AstaEmoBadge({ asta, emo }) {
  const pct = Math.round(((asta.mood_score||0)+1)/2*100);
  return (
    <div style={{display:"flex",alignItems:"center",gap:6,padding:"4px 10px",borderRadius:99,background:`${emo.color}14`,border:`1px solid ${emo.color}30`,fontSize:11,fontFamily:"var(--mono)",color:emo.color,transition:"all .4s"}} title={`Mood: ${asta.mood} | Affection: ${(asta.affection_level||0.7).toFixed(2)}`}>
      <span style={{fontSize:14}}>{emo.emoji}</span>
      <span>{emo.label}</span>
      <div style={{width:32,height:3,background:"var(--border)",borderRadius:99,overflow:"hidden"}}>
        <div className="bar-fill" style={{height:"100%",width:`${pct}%`,background:emo.color,borderRadius:99}}/>
      </div>
    </div>
  );
}

function UserEmoBadge({ user, emo }) {
  return (
    <div style={{display:"flex",alignItems:"center",gap:5,padding:"4px 10px",borderRadius:99,background:"var(--surface2)",border:"1px solid var(--border)",fontSize:11,fontFamily:"var(--mono)",color:"var(--muted)"}} title="Emosi user">
      <span style={{color:emo.color}}>{emo.emoji}</span>
      <span>{user.user_emotion}</span>
    </div>
  );
}

function Bubble({ msg, isStreaming }) {
  const isUser = msg.role === "user";
  return (
    <div style={{display:"flex",justifyContent:isUser?"flex-end":"flex-start",padding:"3px 0",animation:`${isUser?"slideR":"slideL"} .28s ease`}}>
      <div style={{display:"flex",flexDirection:"column",alignItems:isUser?"flex-end":"flex-start",maxWidth:"68%",gap:6}}>
        {!isUser && msg.webSearch && <WebSearchSubBubble webSearch={msg.webSearch} />}
        <div style={{padding:"11px 16px",borderRadius:isUser?"16px 4px 16px 16px":"4px 16px 16px 16px",background:isUser?"var(--asta)":"var(--surface)",color:isUser?"#f5f0eb":"var(--text)",fontSize:13,lineHeight:1.65,boxShadow:"var(--shadow)",border:isUser?"none":"1px solid var(--border)",wordBreak:"break-word",whiteSpace:"pre-wrap",textAlign:"left",width:"100%"}}>
          {isStreaming && msg.tokens
            ? <>{msg.tokens.map(t=><span key={t.id} className="stream-token">{t.text}</span>)}<Cursor/></>
            : <>{msg.content}{isStreaming&&<Cursor/>}</>
          }
        </div>
      </div>
    </div>
  );
}

function WebSearchSubBubble({ webSearch }) {
  const [collapsed, setCollapsed] = useState(false);
  return (
    <div style={{width:"94%",marginLeft:0,border:"1px solid var(--border)",borderRadius:"12px 12px 12px 4px",background:"var(--surface)",boxShadow:"var(--shadow)",overflow:"hidden",animation:"pullOut .25s ease",position:"relative"}}>
      <div style={{position:"absolute",left:10,bottom:-8,width:14,height:14,background:"var(--surface)",borderRight:"1px solid var(--border)",borderBottom:"1px solid var(--border)",transform:"rotate(45deg)"}}/>
      <button onClick={()=>setCollapsed(p=>!p)} style={{width:"100%",display:"flex",alignItems:"center",justifyContent:"space-between",padding:"0px 11px",border:"none",background:"transparent",cursor:"pointer",fontSize:11,fontFamily:"var(--mono)",color:"var(--muted)",position:"relative",zIndex:1}}>
        <span style={{display:"flex",alignItems:"center",gap:6,textAlign:"left"}}>
          <span style={{color:"var(--accent)",fontSize:18, marginBottom:"2px"}}>⌕</span>
          <span style={{color:"var(--text)",fontWeight:500}}>Web search: {webSearch.query}</span>
        </span>
        <span style={{color:"var(--accent)",fontSize:25, marginBottom:"2px"}}>{collapsed ? "▾" : "▴"}</span>
      </button>
      <div style={{maxHeight:collapsed?0:160,overflow:"hidden",transition:"max-height var(--ease)",position:"relative",zIndex:1}}>
        <div style={{maxHeight:140,overflowY:"auto",padding:"10px 13px 10px",fontSize:12,textAlign:"left",lineHeight:1.55,color:"var(--text)",whiteSpace:"pre-wrap",wordBreak:"break-word",borderTop:"1px dashed var(--border)"}}>
          {webSearch.result}
        </div>
      </div>
    </div>
  );
}

const Cursor = () => (
  <span style={{display:"inline-block",width:7,height:14,background:"var(--accent)",borderRadius:2,marginLeft:2,verticalAlign:"text-bottom",animation:"blink .8s step-end infinite"}}/>
);

function ThinkingBubble() {
  return (
    <div style={{display:"flex",padding:"3px 0",animation:"fadeIn .3s ease"}}>
      <div style={{padding:"13px 18px",borderRadius:"4px 16px 16px 16px",background:"var(--surface)",border:"1px solid var(--border)",boxShadow:"var(--shadow)",display:"flex",gap:6,alignItems:"center"}}>
        {[0,.18,.36].map(d=><div key={d} style={{width:6,height:6,borderRadius:"50%",background:"var(--accent)",animation:`pulse 1.2s ease-in-out ${d}s infinite`}}/>)}
      </div>
    </div>
  );
}

function TopBtn({ active, onClick, icon, label, color, onMouseEnter, onMouseLeave }) {
  const activeBg = color || "var(--asta)";
  const isDark = document.documentElement.classList.contains("dark");
  
  return (
    <button 
      onClick={onClick} 
      onMouseEnter={onMouseEnter}
      onMouseLeave={onMouseLeave}
      style={{
        display:"flex",alignItems:"center",gap:6,padding:"6px 14px",borderRadius:99,
        background:active ? activeBg : "var(--surface)",
        color:active ? (isDark ? "#1a1816" : "#f5f0eb") : (color && !active ? color : "var(--muted)"),
        border:`1px solid ${active ? activeBg : (color ? color : "var(--border)")}`,
        fontSize:12,fontFamily:"var(--font)",fontWeight:500,cursor:"pointer",transition:"all var(--ease)",
        WebkitAppRegion:"no-drag"
      }}
    >
      {icon} {label}
    </button>
  );
}

function SidePanel({ visible, side, title, icon, width=260, noPadding=false, children }) {
  return (
    <div style={{width:visible?width:0,minWidth:visible?width:0,overflow:"hidden",flexShrink:0,transition:"width .3s cubic-bezier(0.4,0,0.2,1),min-width .3s cubic-bezier(0.4,0,0.2,1)"}}>
      <div style={{width,height:"100%",background:"var(--surface)",borderLeft:side==="right"?"1px solid var(--border)":"none",borderRight:side==="left"?"1px solid var(--border)":"none",display:"flex",flexDirection:"column",opacity:visible?1:0,transition:"opacity .3s"}}>
        <div style={{padding:"14px 16px 10px",borderBottom:"1px solid var(--border)",display:"flex",alignItems:"center",gap:8,fontSize:11,fontWeight:600,letterSpacing:"0.06em",color:"var(--muted)",textTransform:"uppercase",flexShrink:0}}>
          {icon} {title}
        </div>
        <div style={{flex:1,overflowY:"auto",padding:noPadding?0:14}}>
          {children}
        </div>
      </div>
    </div>
  );
}

function ThoughtPanel({ thought, thinking, modelInfo }) {
  if (thinking) return (
    <div style={{display:"flex",flexDirection:"column",alignItems:"center",paddingTop:40}}>
      <div style={{width:18,height:18,borderRadius:"50%",border:"2px solid var(--border)",borderTop:"2px solid var(--accent)",animation:"spin .8s linear infinite",marginBottom:10}}/>
      <span style={{fontSize:11,color:"var(--muted)"}}>{modelInfo.dual_model?`Berpikir… (${modelInfo.thought_model})`:"Berpikir…"}</span>
    </div>
  );

  if (!thought) return <div style={{color:"var(--muted)",fontSize:12,textAlign:"center",paddingTop:40}}>Belum ada thought</div>;

  const mi = thought.model_info || modelInfo;
  const steps = [
    { num:"1", label:"PERCEPTION", rows:[
      { k:"Topic",     v: thought.topic    },
      { k:"Sentiment", v: thought.sentiment },
      { k:"Urgency",   v: thought.urgency   },
    ]},
    { num:"2", label:"SELF-CHECK", color:"var(--rose)", rows:[
      { k:"Emosi Asta", v: thought.asta_emotion, color: getEmo(thought.asta_emotion).color },
      { k:"Trigger",    v: thought.asta_trigger },
      { k:"Ekspresikan",v: thought.should_express ? "Ya" : "Tidak" },
    ]},
    { num:"3", label:"MEMORY", color:"var(--blue)", rows:[
      { k:"Need Search", v: thought.need_search ? "✓ Ya" : "✗ Tidak", mono:true },
      thought.search_query && { k:"Query", v: thought.search_query, mono:true },
      { k:"Recall", v: thought.recall_topic || "–" },
      { k:"Use Memory", v: thought.use_memory ? "✓ Ya" : "✗ Tidak", mono:true },
    ].filter(Boolean)},
    { num:"4", label:"DECISION", color:"var(--green)", rows:[
      { k:"Tone",  v: thought.tone  },
      { k:"Style", v: thought.response_style || "normal" },
    ].filter(Boolean)},
  ];

  return (
    <div style={{display:"flex",flexDirection:"column",gap:8,animation:"fadeIn .3s ease"}}>
      {mi.dual_model && (
        <div style={{padding:"6px 9px",borderRadius:"var(--rs)",border:"1px solid var(--border)",marginBottom:2}}>
          <div style={S.cardLabel}>Pipeline</div>
          <div style={{display:"center",gap:6,marginTop:3,flexWrap:"wrap"}}>
            <span style={{padding:"3px 7px",marginRight:5,borderRadius:99,fontSize:10,fontFamily:"var(--mono)",fontWeight:600,background:"#7a9ec722",color:"#7a9ec7",border:"1px solid #7a9ec733"}}>⟡ {mi.thought_model}</span>
            <span style={{fontSize:10,color:"var(--muted)",alignSelf:"center"}}>→</span>
            <span style={{padding:"3px 7px",marginLeft:5,borderRadius:99,fontSize:10,fontFamily:"var(--mono)",fontWeight:600,background:"var(--accent)22",color:"var(--accent)",border:"1px solid var(--accent)33"}}>↑ {mi.response_model}</span>
          </div>
        </div>
      )}

      {steps.map(step => (
        <div key={step.num} style={{borderRadius:"var(--rs)",border:`1px solid ${step.color||"var(--border)"}33`,overflow:"hidden"}}>
          <div style={{padding:"5px 10px",background:`${step.color||"var(--accent)"}12`,borderBottom:`1px solid ${step.color||"var(--border)"}18`,fontSize:10.5,fontWeight:700,letterSpacing:"0.08em",color:step.color||"var(--muted)",fontFamily:"var(--mono)",textAlign:"left"}}>
            S{step.num} · {step.label}
          </div>
          <div style={{padding:"0px 10px",display:"flex",flexDirection:"column"}}>
            {step.rows.map((row,i) => row && (
              <div key={i} style={{display:"grid",gridTemplateColumns:"85px 1fr",gap:10,fontSize:11,alignItems:"start",textAlign:"left"}}>
                <span style={{color:"var(--muted)",fontWeight:500}}>{row.k}</span>
                <span style={{fontFamily:row.mono?"var(--mono)":"var(--font)",color:row.color||"var(--text)",textAlign:"left",wordBreak:"break-word",lineHeight:2.5}}>{row.v||"–"}</span>
              </div>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}

function SelfPanel({ selfModel, astaEmotion, onReflect }) {
  if (!selfModel) return <div style={{color:"var(--muted)",fontSize:12,textAlign:"center",paddingTop:40}}>Memuat…</div>;

  const emo = getEmo(astaEmotion.current_emotion || astaEmotion.mood);
  const moodPct = Math.round(((astaEmotion.mood_score||0)+1)/2*100);
  const affPct  = Math.round((astaEmotion.affection_level||0.7)*100);
  const engPct  = Math.round((astaEmotion.energy_level||0.8)*100);

  return (
    <div style={{display:"flex",flexDirection:"column",gap:14,animation:"fadeIn .3s ease"}}>

      {/* Emotional state */}
      <div>
        <div style={S.sectionTitle}>Kondisi Emosional</div>
        <div style={{marginTop:7,padding:"10px 12px",borderRadius:"var(--rs)",background:`${emo.color}0e`,border:`1px solid ${emo.color}25`}}>
          <div style={{display:"flex",alignItems:"center",gap:8,marginBottom:10}}>
            <span style={{fontSize:24}}>{emo.emoji}</span>
            <div>
              <div style={{fontSize:13,fontWeight:600,marginLeft:5,textAlign:"left",color:emo.color}}>{emo.label}</div>
              <div style={{fontSize:11,marginLeft:5,textAlign:"left",color:"var(--muted)",fontFamily:"var(--mono)"}}>{astaEmotion.mood}</div>
            </div>
          </div>
          {[
            { label:"Mood",      pct:moodPct,  color:emo.color,    val:`${astaEmotion.mood_score>=0?"+":""}${(astaEmotion.mood_score||0).toFixed(2)}` },
            { label:"Affection", pct:affPct,   color:"var(--rose)", val:`${affPct}%` },
            { label:"Energy",    pct:engPct,   color:"var(--green)",val:`${engPct}%` },
          ].map(b => (
            <div key={b.label} style={{marginBottom:6}}>
              <div style={{display:"flex",justifyContent:"space-between",fontSize:10,color:"var(--muted)",marginBottom:2}}>
                <span>{b.label}</span><span style={{fontFamily:"var(--mono)",color:b.color}}>{b.val}</span>
              </div>
              <div style={{height:4,background:"var(--border)",borderRadius:99,overflow:"hidden"}}>
                <div className="bar-fill" style={{height:"100%",width:`${b.pct}%`,background:b.color,borderRadius:99}}/>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Nilai inti */}
      {selfModel.identity?.nilai_inti?.length > 0 && (
        <div>
          <div style={S.sectionTitle}>Nilai Inti</div>
          <div style={{display:"flex",flexDirection:"column",gap:4,marginTop:6}}>
            {selfModel.identity.nilai_inti.map((v,i) => (
              <div key={i} style={{fontSize:11,padding:"4px 8px",borderRadius:6,background:"var(--surface2)",border:"1px solid var(--border)",color:"var(--text)"}}>
                ◦ {v}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Kenangan diri */}
      {selfModel.memories_of_self?.length > 0 && (
        <div>
          <div style={S.sectionTitle}>Kenangan Diri</div>
          <div style={{display:"flex",flexDirection:"column",gap:4,marginTop:6}}>
            {selfModel.memories_of_self.map((m,i) => (
              <div key={i} style={{fontSize:11,padding:"5px 8px",borderRadius:6,background:"var(--surface2)",border:"1px solid var(--border)"}}>
                <div style={{color:"var(--muted)",fontFamily:"var(--mono)",fontSize:9,marginBottom:2}}>{m.timestamp?.slice(0,10)}</div>
                <div>{m.content}</div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Refleksi terakhir */}
      {selfModel.last_reflection && (
        <div>
          <div style={S.sectionTitle}>Refleksi Terakhir</div>
          <div style={{marginTop:6,padding:"8px 10px",borderRadius:"var(--rs)",background:"var(--surface2)",border:"1px solid var(--border)",fontSize:12,lineHeight:1.6}}>
            {selfModel.last_reflection.summary}
          </div>
          {selfModel.last_reflection.growth_note && (
            <div style={{marginTop:4,fontSize:11,color:"var(--accent)",fontStyle:"italic",padding:"0 2px"}}>
              ✦ {selfModel.last_reflection.growth_note}
            </div>
          )}
          <div style={{fontSize:10,color:"var(--muted)",marginTop:4,fontFamily:"var(--mono)"}}>
            {selfModel.reflection_count} refleksi tersimpan
          </div>
        </div>
      )}

      {/* Growth log */}
      {selfModel.growth_log?.length > 0 && (
        <div>
          <div style={S.sectionTitle}>Growth Log</div>
          <div style={{display:"flex",flexDirection:"column",gap:3,marginTop:6}}>
            {selfModel.growth_log.slice(-3).map((g,i) => (
              <div key={i} style={{fontSize:11,padding:"4px 8px",borderRadius:6,background:"var(--surface2)",border:"1px solid var(--border)"}}>
                <span style={{color:"var(--muted)",fontFamily:"var(--mono)",fontSize:9}}>{g.timestamp?.slice(0,10)} </span>
                {g.entry}
              </div>
            ))}
          </div>
        </div>
      )}

      <button onClick={onReflect} style={{...S.refreshBtn,borderColor:"var(--accent)44",color:"var(--accent)"}}>
        ✦ Refleksi Sekarang
      </button>
    </div>
  );
}

function MemoryPanel({ memory, onRefresh }) {
  if (!memory) return <div style={{color:"var(--muted)",fontSize:13,textAlign:"center",paddingTop:40}}>Memuat…</div>;
  return (
    <div style={{display:"flex",flexDirection:"column",gap:16,animation:"fadeIn .3s ease"}}>
      {memory.profile?.preferensi?.length > 0 && (
        <div>
          <div style={S.sectionTitle}>Preferensi</div>
          <div style={{display:"flex",flexWrap:"wrap",gap:5,marginTop:7}}>
            {memory.profile.preferensi.map((p,i) => <span key={i} style={S.tag}>{p}</span>)}
          </div>
        </div>
      )}
      {memory.recent_facts && (
        <div>
          <div style={S.sectionTitle}>Fakta Terbaru</div>
          <div style={{fontFamily:"var(--mono)",fontSize:11,color:"var(--muted)",lineHeight:1.8,marginTop:7,whiteSpace:"pre-wrap"}}>{memory.recent_facts}</div>
        </div>
      )}
      {memory.core && (
        <div>
          <div style={S.sectionTitle}>Core Summary</div>
          <div style={{fontSize:12,lineHeight:1.7,marginTop:7,padding:"10px 12px",background:"var(--surface2)",borderRadius:"var(--rs)"}}>{memory.core}</div>
        </div>
      )}
      {memory.sessions?.length > 0 && (
        <div>
          <div style={S.sectionTitle}>Sesi Tersimpan</div>
          <div style={{display:"flex",flexDirection:"column",gap:6,marginTop:7}}>
            {memory.sessions.map((s,i) => (
              <div key={i} style={{padding:"6px 9px",borderRadius:"var(--rs)",background:"var(--surface2)",border:"1px solid var(--border)"}}>
                <div style={{fontSize:10,color:"var(--muted)",fontFamily:"var(--mono)"}}>{new Date(s.timestamp).toLocaleString("id-ID",{dateStyle:"short",timeStyle:"short"})} · {s.facts} fakta</div>
                <div style={{fontSize:11,marginTop:3,lineHeight:1.5}}>{s.preview}</div>
              </div>
            ))}
          </div>
        </div>
      )}
      <button onClick={onRefresh} style={S.refreshBtn}>↻ Refresh</button>
    </div>
  );
}

const S = {
  root:      { display:"flex",flexDirection:"column",width:"100%",height:"100vh",overflow:"hidden",background:"var(--bg)" },
  topBar:    { display:"flex",alignItems:"center",gap:8,padding:"10px 20px",paddingTop:"35px",flexShrink:0,borderBottom:"1px solid var(--border)",background:"var(--bg)",WebkitAppRegion:"drag" },
  layout:    { flex:1,display:"flex",overflow:"hidden",minHeight:0 },
  chatCol:   { flex:1,display:"flex",flexDirection:"column",minWidth:0 },
  header:    { display:"flex",alignItems:"center",justifyContent:"space-between",padding:"14px 24px",flexShrink:0,borderBottom:"1px solid var(--border)",background:"var(--surface)" },
  hLeft:     { display:"flex",alignItems:"center" },
  hName:     { fontSize:17,textAlign:"left",fontWeight:600,letterSpacing:"-0.01em",lineHeight:1.4 },
  hSub:      { fontSize:12,textAlign:"left",color:"var(--muted)",marginTop:1,fontFamily:"var(--mono)",lineHeight:1.3 },
  hRight:    { display:"flex",alignItems:"center",gap:10 },
  msgList:   { flex:1,overflowY:"auto",padding:"24px 32px",display:"flex",flexDirection:"column",gap:4,minHeight:0 },
  noteBubbleFixed: { position:"absolute",left:395,top:8,width:320,padding:"14px 18px",background:"var(--surface)",border:"1.5px solid var(--green)",borderRadius:"16px",boxShadow:"0 10px 40px rgba(0,0,0,0.15)",zIndex:1000,fontSize:12,lineHeight:1.55,color:"var(--text)",fontStyle:"italic",textAlign:"left" },
  statsBox:  { position:"absolute",right:20,top:65,width:200,padding:"16px",background:"var(--surface)",border:"1px solid var(--border)",borderRadius:"16px",boxShadow:"var(--shadow)",zIndex:1000,animation:"pullOut .3s ease" },
  empty:     { display:"flex",flex:1,flexDirection:"column",alignItems:"center",justifyContent:"center",opacity:0.55,animation:"fadeIn .6s ease" },
  inputWrap: { padding:"14px 24px 18px",flexShrink:0,borderTop:"1px solid var(--border)",background:"var(--surface)" },
  inputRow:  { display:"flex",gap:12,alignItems:"flex-end" },
  textarea:  { flex:1,resize:"none",padding:"12px 16px",borderRadius:"var(--r)",border:"1.5px solid var(--border)",background:"var(--surface2)",fontSize:15,fontFamily:"var(--font)",color:"var(--text)",lineHeight:1.6,outline:"none",maxHeight:140,overflowY:"auto" },
  sendBtn:   { width:50,height:50,borderRadius:"50%",background:"var(--asta)",color:"#f5f0eb",border:"none",fontSize:18,cursor:"pointer",transition:"all var(--ease)",flexShrink:0,display:"flex",alignItems:"center",justifyContent:"center",fontWeight:600 },
  hint:      { fontSize:11,color:"var(--muted)",marginTop:7,fontFamily:"var(--mono)" },
  saveBtn:   { width:36,height:36,borderRadius:"50%",background:"transparent",border:"1px solid var(--border)",color:"var(--muted)",fontSize:14,cursor:"pointer",display:"flex",alignItems:"center",justifyContent:"center" },
  modelBadge:{ display:"flex",alignItems:"center",gap:6,padding:"4px 10px",borderRadius:99,background:"var(--surface2)",border:"1px solid var(--border)",fontSize:11,fontFamily:"var(--mono)",fontWeight:500 },
  cardLabel: { fontSize:11,color:"var(--muted)",textTransform:"uppercase",letterSpacing:"0.06em",marginBottom:2,fontWeight:600 },
  sectionTitle:{ fontSize:11,fontWeight:600,color:"var(--muted)",textTransform:"uppercase",marginBottom:3,letterSpacing:"0.06em" },
  tag:       { padding:"4px 11px",borderRadius:99,background:"var(--tag-bg)",fontSize:12,color:"var(--text)",fontWeight:500 },
  refreshBtn:{ width:"100%",padding:"9px",borderRadius:"var(--rs)",border:"1px solid var(--border)",background:"transparent",cursor:"pointer",fontSize:13,color:"var(--muted)",fontFamily:"var(--font)" },
};
