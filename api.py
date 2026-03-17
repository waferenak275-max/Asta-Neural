"""
api.py — FastAPI + WebSocket backend for Asta AI
"""

import asyncio
import json
import threading

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

_chat_manager  = None
_hybrid_memory = None
_init_lock     = threading.Lock()
_initialized   = False


def _initialize():
    global _chat_manager, _hybrid_memory, _initialized
    if _initialized:
        return
    with _init_lock:
        if _initialized:
            return
        from config import load_config
        from engine.model import load_model
        from engine.memory import get_hybrid_memory, get_identity

        cfg       = load_config()
        user_name = get_identity("nama_user") or "Aditiya"
        cfg["_user_name"] = user_name

        chat_manager       = load_model(cfg)
        hybrid_mem         = get_hybrid_memory()
        chat_manager.hybrid_memory = hybrid_mem

        _chat_manager  = chat_manager
        _hybrid_memory = hybrid_mem
        _initialized   = True


app = FastAPI(title="Asta AI API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _initialize)


@app.get("/status")
async def status():
    if not _initialized:
        return {"ready": False}
    import numpy as np
    ep_count = len([
        s for s in _hybrid_memory.episodic.data
        if not np.allclose(np.array(s.get("embedding", [0])[:5]), 0.0)
    ])
    sep = _chat_manager.llama_thought is not _chat_manager.llama
    return {
        "ready":             True,
        "model":             _chat_manager.cfg.get("model_choice", "?"),
        "user_name":         _chat_manager._user_name_cache,
        "episodic_sessions": ep_count,
        "dual_model":        sep,
        "thought_model":     "3B" if sep else "shared",
        "response_model":    "8B" if _chat_manager.cfg.get("model_choice", "2") == "2" else "3B",
    }


@app.get("/memory")
async def get_memory():
    if not _initialized:
        return JSONResponse({"error": "not initialized"}, status_code=503)
    core_text    = _hybrid_memory.core.get_context_text()
    recent_facts = _hybrid_memory.episodic.get_recent_facts_text(n_sessions=3, max_facts=10)
    profile      = _hybrid_memory.core.get_profile()
    previews     = []
    for s in _hybrid_memory.episodic.get_last_n(5):
        conv       = s.get("conversation", [])
        first_user = next((m["content"] for m in conv if m["role"] == "user"), "")
        previews.append({
            "timestamp": s.get("timestamp", ""),
            "preview":   first_user[:80],
            "facts":     len(s.get("key_facts", [])),
        })
    return {
        "core":         core_text,
        "recent_facts": recent_facts,
        "profile":      profile,
        "sessions":     previews,
    }


@app.get("/self")
async def get_self_model():
    if not _initialized:
        return JSONResponse({"error": "not initialized"}, status_code=503)
    sm   = _chat_manager.self_model
    comb = _chat_manager.emotion_manager.get_combined()
    refs = sm.data.get("reflection_history", [])
    return {
        "identity":          sm.data.get("identity", {}),
        "emotional_state":   comb["asta"],
        "preferences":       sm.data.get("preferences", {}),
        "learned_behaviors": sm.data.get("learned_behaviors", {}),
        "memories_of_self":  sm.data.get("memories_of_self", [])[-5:],
        "growth_log":        sm.data.get("growth_log", [])[-5:],
        "last_reflection":   refs[-1] if refs else None,
        "reflection_count":  len(refs),
    }


@app.get("/emotion")
async def get_emotion():
    if not _initialized:
        return JSONResponse({"error": "not initialized"}, status_code=503)
    return _chat_manager.emotion_manager.get_combined()


@app.get("/config")
async def get_config():
    if not _initialized:
        return JSONResponse({"error": "not initialized"}, status_code=503)
    sep = _chat_manager.llama_thought is not _chat_manager.llama
    return {
        "internal_thought_enabled": _chat_manager.cfg.get("internal_thought_enabled", True),
        "web_search_enabled":       _chat_manager.cfg.get("web_search_enabled", True),
        "dual_model":               sep,
        "thought_model":            "3B" if sep else "shared",
        "response_model":           "8B" if _chat_manager.cfg.get("model_choice", "2") == "2" else "3B",
    }


@app.post("/config/thought")
async def toggle_thought():
    if not _initialized:
        return JSONResponse({"error": "not initialized"}, status_code=503)
    from config import save_config
    cur = _chat_manager.cfg.get("internal_thought_enabled", True)
    _chat_manager.cfg["internal_thought_enabled"] = not cur
    save_config(_chat_manager.cfg)
    return {"internal_thought_enabled": _chat_manager.cfg["internal_thought_enabled"]}


@app.post("/save")
async def save_session():
    if not _initialized:
        return JSONResponse({"error": "not initialized"}, status_code=503)
    from engine.memory import add_episodic
    conv = _chat_manager._clean_conversation()
    if conv:
        _hybrid_memory.extract_and_save_preferences(conv)
        add_episodic(conv)
        session_text = _chat_manager.get_session_text()
        if session_text:
            _hybrid_memory.update_core_async(
                llm_callable=_chat_manager.llama.create_completion,
                current_session_text=session_text,
            )
    return {"saved": len(conv)}


@app.post("/reflect")
async def trigger_reflection():
    if not _initialized:
        return JSONResponse({"error": "not initialized"}, status_code=503)
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _chat_manager.run_exit_reflection)
    return {"status": "done"}


@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    await websocket.accept()
    loop = asyncio.get_event_loop()

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                data       = json.loads(raw)
                user_input = data.get("message", "").strip()
                if not user_input:
                    continue
            except json.JSONDecodeError:
                user_input = raw.strip()

            if not _initialized:
                await websocket.send_text(json.dumps({"type": "error", "text": "Model belum siap."}))
                continue

            await websocket.send_text(json.dumps({"type": "thinking_start"}))

            thought_holder = {}
            emotion_holder = {}
            asta_holder    = {}
            chunk_queue    = asyncio.Queue()

            def _run_chat():
                import datetime
                from engine.thought import run_thought_pass, extract_recent_context
                from engine.web_tools import search_and_summarize

                cm  = _chat_manager
                now = datetime.datetime.now()
                ts  = now.strftime("%A, %d %B %Y %H:%M WIB")

                memory_ctx = cm._get_memory_context(query=user_input, recall_topic="")
                recent_ctx = extract_recent_context(cm.conversation_history, n=2)
                em_dict    = cm.emotion_manager.update(user_input, recent_context=recent_ctx)

                thought = {
                    "topic": "", "sentiment": "netral", "urgency": "normal",
                    "asta_emotion": "netral", "asta_trigger": "", "should_express": False,
                    "need_search": False, "search_query": "", "recall_topic": "", "use_memory": False,
                    "recall_source": "none", "tone": "romantic", "note": "", "raw": "",
                }

                if cm.cfg.get("internal_thought_enabled", True):
                    cm._maybe_reset_thought_kv()
                    thought = run_thought_pass(
                        llm=cm.llama_thought,
                        user_input=user_input,
                        memory_context=memory_ctx,
                        recent_context=recent_ctx,
                        web_search_enabled=cm.cfg.get("web_search_enabled", True),
                        max_tokens=50,
                        user_name=cm._user_name_cache,
                        emotion_state=(
                            f"emosi={em_dict['user_emotion']}; "
                            f"intensitas={em_dict['intensity']}; "
                            f"tren={em_dict['trend']}"
                        ),
                        asta_state=cm.emotion_manager.get_asta_dict(),
                    )
                    em_dict = cm.emotion_manager.refine_with_thought(thought)

                cm.emotion_manager.update_asta_emotion(thought)
                cm.self_model.sync_emotion(cm.emotion_manager.get_asta_dict())

                thought_holder["thought"] = thought
                emotion_holder["emotion"] = em_dict
                asta_holder["asta"]       = cm.emotion_manager.get_asta_dict()

                emotion_guidance = cm.emotion_manager.build_prompt_context()

                # Supplemental recall
                recall_topic = thought.get("recall_topic", "")
                should_recall = bool(thought.get("use_memory") or recall_topic)
                if should_recall and cm.hybrid_memory:
                    target_topic = (recall_topic or thought.get("topic") or user_input[:60]).strip()
                    if target_topic and target_topic.lower() not in ("kosong", "-"):
                        supplemental = cm.hybrid_memory.episodic.search_by_facts(target_topic, top_k=1)
                        if supplemental:
                            s    = supplemental[0]
                            conv = s.get("conversation", [])
                            kws  = [w for w in target_topic.lower().split() if len(w) > 2]
                            lines = []
                            for i, msg in enumerate(conv):
                                if msg.get("role") == "user":
                                    content = msg.get("content", "")
                                    if any(kw in content.lower() for kw in kws):
                                        lines.append(f"Aditiya: {content[:100]}")
                                        if i+1 < len(conv) and conv[i+1].get("role") == "assistant":
                                            lines.append(f"Asta: {conv[i+1]['content'][:100]}")
                                        if len(lines) >= 4:
                                            break
                            if lines:
                                recall_block = f"[Ingatan: '{target_topic}']\n" + "\n".join(lines)
                                if recall_block not in memory_ctx:
                                    memory_ctx = (f"{memory_ctx}\n\n{recall_block}".strip() if memory_ctx else recall_block)

                web_result = ""
                if (cm.cfg.get("web_search_enabled", True)
                        and thought["need_search"]
                        and thought.get("search_query")):
                    web_result = search_and_summarize(
                        thought["search_query"], max_results=2, timeout=5)
                    if not web_result:
                        web_result = "[INFO] Web search gagal."
                thought["web_result"] = web_result

                # Build messages dengan strategi KV cache baru
                static_system   = {"role": "system", "content": cm.system_identity}
                dynamic_context = cm._build_dynamic_context(
                    timestamp_str=ts,
                    memory_ctx=memory_ctx,
                    web_result=web_result,
                    emotion_guidance=emotion_guidance,
                    thought_note=thought.get("note", ""),
                )

                # Tambah user input ke history (bersih)
                cm.conversation_history.append({"role": "user", "content": user_input})

                messages_to_send, _ = cm.budget_manager.build_messages(
                    system_identity=static_system,
                    memory_messages=[],
                    conversation_history=cm.conversation_history,
                    dynamic_context=dynamic_context,
                )

                sep = cm.llama_thought is not cm.llama
                loop.call_soon_threadsafe(chunk_queue.put_nowait, {"type": "thought_ready"})

                response_stream = cm.llama.create_chat_completion(
                    messages=messages_to_send,
                    max_tokens=128,
                    temperature=0.7,
                    top_p=0.85,
                    top_k=60,
                    stop=["<|im_end|>", "<|endoftext|>"],
                    stream=True,
                )

                full_response = ""
                for chunk in response_stream:
                    delta = chunk["choices"][0]["delta"]
                    if "content" in delta:
                        text = delta["content"]
                        full_response += text
                        loop.call_soon_threadsafe(
                            chunk_queue.put_nowait, {"type": "chunk", "text": text})

                cm.conversation_history.append({"role": "assistant", "content": full_response})
                loop.call_soon_threadsafe(chunk_queue.put_nowait, {"type": "done"})

            import concurrent.futures
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            future   = loop.run_in_executor(executor, _run_chat)

            while True:
                item = await chunk_queue.get()
                if item["type"] == "thought_ready":
                    thought = thought_holder.get("thought", {})
                    emotion = emotion_holder.get("emotion", {})
                    asta    = asta_holder.get("asta", {})
                    sep     = _chat_manager.llama_thought is not _chat_manager.llama
                    await websocket.send_text(json.dumps({
                        "type": "thought",
                        "data": {
                            "topic":          thought.get("topic", ""),
                            "sentiment":      thought.get("sentiment", ""),
                            "urgency":        thought.get("urgency", ""),
                            "asta_emotion":   thought.get("asta_emotion", ""),
                            "asta_trigger":   thought.get("asta_trigger", ""),
                            "should_express": thought.get("should_express", False),
                            "need_search":    thought.get("need_search", False),
                            "search_query":   thought.get("search_query", ""),
                            "web_result":     thought.get("web_result", ""),
                            "recall_topic":   thought.get("recall_topic", ""),
                            "use_memory":     thought.get("use_memory", False),
                            "recall_source":  thought.get("recall_source", "none"),
                            "tone":           thought.get("tone", ""),
                            "note":           thought.get("note", ""),
                            "response_style": thought.get("response_style", ""),
                            "emotion":        emotion,
                            "asta_state":     asta,
                            "model_info": {
                                "dual_model":     sep,
                                "thought_model":  "3B" if sep else "shared",
                                "response_model": "8B" if _chat_manager.cfg.get("model_choice","2")=="2" else "3B",
                            },
                        }
                    }))
                    await websocket.send_text(json.dumps({"type": "stream_start"}))

                elif item["type"] == "chunk":
                    await websocket.send_text(json.dumps({"type": "chunk", "text": item["text"]}))

                elif item["type"] == "done":
                    await websocket.send_text(json.dumps({"type": "stream_end"}))
                    break

            await future

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_text(json.dumps({"type": "error", "text": str(e)}))
        except Exception:
            pass
