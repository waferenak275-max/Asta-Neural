"""
core.py — Entry point Asta (versi dioptimasi).

Perubahan utama dari versi sebelumnya:
  ✓ Startup cepat: config dibaca dari file, tidak ada wizard blocking
  ✓ Memory loading non-blocking dengan progress yang jelas
  ✓ Dual-pass inference (internal thought + response)
  ✓ Web search terintegrasi, hanya aktif jika dibutuhkan
  ✓ Token budget yang ketat dan benar
  ✓ Core memory update di background (tidak blocking saat exit)
  ✓ Episodic memory: key facts extraction tanpa LLM (cepat)
  ✓ LLM summarization hanya sebagai background update opsional

Jalankan:
  python core.py           — mode normal
  python core.py --setup   — reset konfigurasi
  python core.py --debug   — tampilkan thought pass
"""

import sys
import re
import argparse
from pathlib import Path

from config import load_config, save_config, setup_wizard
from engine.model import load_model
from engine.memory import (
    remember_identity,
    get_identity,
    add_episodic,
    get_hybrid_memory,
)
from utils.spinner import Spinner


# ─── Argparse ─────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Asta AI Companion")
parser.add_argument("--setup", action="store_true", help="Reset konfigurasi")
parser.add_argument("--debug", action="store_true", help="Tampilkan internal thought")
args = parser.parse_args()


# ─── Config ───────────────────────────────────────────────────────────────────

cfg = load_config()

# Jalankan setup wizard jika pertama kali atau diminta
if args.setup or not Path("config.json").exists():
    cfg = setup_wizard(cfg)


# ─── User Name ────────────────────────────────────────────────────────────────

def get_or_set_user_name() -> str:
    current = get_identity("nama_user")
    if current:
        # Hanya tanya ganti nama, tidak tanya mode/model/dll
        print(f"\nHalo! Aku ingat kamu: {current}")
        change = input("Ganti nama? (y/n, default = n): ").strip().lower()
        if change == "y":
            new_name = input("Nama baru: ").strip().capitalize()
            if new_name:
                remember_identity("nama_user", new_name)
                return new_name
        return current
    else:
        new_name = input("\nNama kamu siapa? (kosong = 'Aditiya'): ").strip().capitalize()
        name = new_name or "Aditiya"
        remember_identity("nama_user", name)
        return name

user_name = get_or_set_user_name()


# ─── Load Model (Non-Blocking, Sekali) ────────────────────────────────────────

print(f"\n[Startup] Memuat model...")
chat_manager = load_model(cfg)


# ─── Kaitkan Hybrid Memory ke ChatManager ─────────────────────────────────────

hybrid_mem = get_hybrid_memory()
chat_manager.hybrid_memory = hybrid_mem
chat_manager.debug_thought = args.debug

# Injeksi nama user ke system identity
chat_manager.system_identity += f"\n- Nama pengguna: {user_name}."


# ─── Tampilkan Status Memory ──────────────────────────────────────────────────

print("\n[Memory] Status:")
core_text = hybrid_mem.core.get_summary()
ep_count = len(hybrid_mem.episodic.data)
if core_text:
    preview = core_text[:80] + "..." if len(core_text) > 80 else core_text
    print(f"  Core : {preview}")
else:
    print("  Core : (kosong)")
print(f"  Sesi : {ep_count} sesi episodik tersimpan")

recent_facts = hybrid_mem.episodic.get_recent_facts_text(n_sessions=3, max_facts=5)
if recent_facts:
    print(f"  Fakta: {recent_facts[:100]}...")

print()


# ─── Utility ──────────────────────────────────────────────────────────────────

def clean_response(text: str) -> str:
    text = re.sub(
        r"^\s*(Asta|Pengguna)\s*[:]?\s*", "", text.strip(),
        flags=re.IGNORECASE | re.MULTILINE,
    )
    return text.strip()


# ─── Main Loop ────────────────────────────────────────────────────────────────

print("=" * 50)
print("  Asta siap! Ketik 'exit' untuk keluar.")
if args.debug:
    print("  [DEBUG MODE] Internal thought akan ditampilkan.")
print("=" * 50 + "\n")

while True:
    sys.stdout.write("Kamu: ")
    sys.stdout.flush()
    user_input = sys.stdin.readline().strip()

    if not user_input:
        continue

    if user_input.lower() == "exit":
        print("\n[Exit] Menyimpan sesi...")

        # ── Simpan ke Episodic Memory (cepat, key facts extraction) ───────
        # Bangun conversation list dari history
        conversation = [
            {"role": m["role"], "content": m["content"]}
            for m in chat_manager.conversation_history
            if m["content"]
        ]
        add_episodic(conversation)  # key facts otomatis, tanpa LLM
        print("[Exit] Sesi disimpan ke episodic memory.")

        # ── Update Core Memory di Background (tidak blocking) ─────────────
        session_text = chat_manager.get_session_text()
        if session_text:
            print("[Exit] Memperbarui core memory di background...")
            hybrid_mem.update_core_async(
                llm_callable=chat_manager.llama.create_completion,
                current_session_text=session_text,
            )
            print("[Exit] Core memory update berjalan di background.")
            print("       (Selesai beberapa detik setelah program tertutup)")

        print("\nDaa Aditiya! Asta tunggu kamu balik ya~ 💕\n")
        break

    # Perintah khusus
    if user_input.lower() == "!memory":
        ctx = hybrid_mem.get_context(max_chars=800)
        print(f"\n[Memory Context]\n{ctx}\n")
        continue

    if user_input.lower() == "!thought":
        chat_manager.debug_thought = not chat_manager.debug_thought
        status = "ON" if chat_manager.debug_thought else "OFF"
        print(f"[Debug] Internal thought: {status}\n")
        continue

    if user_input.lower() == "!web":
        cfg["web_search_enabled"] = not cfg.get("web_search_enabled", True)
        save_config(cfg)
        chat_manager.cfg = cfg
        status = "ON" if cfg["web_search_enabled"] else "OFF"
        print(f"[Config] Web search: {status}\n")
        continue

    # Chat normal
    try:
        response = chat_manager.chat(user_input)
        clean_response(response)  # sudah diprint oleh streaming
    except Exception as e:
        print(f"[Error] {e}\n")
