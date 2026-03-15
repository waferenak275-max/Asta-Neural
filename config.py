"""
config.py — Menyimpan preferensi startup Asta agar tidak perlu ditanya ulang.
"""

import json
from pathlib import Path

CONFIG_PATH = Path("config.json")

DEFAULT_CONFIG = {
    "model_choice": "2",           # "1" = Sailor2 3B, "2" = Sailor2 8B
    "device": "cpu",               # "cpu" atau "gpu"
    "use_lora": False,
    "lora_n_gpu_layers": 0,
    "memory_mode": "hybrid",       # "hybrid" (baru), "core_memory", "episodic_summarized"
    "web_search_enabled": True,
    "tavily_api_key": "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",          # Daftar gratis (1000/bulan) di https://app.tavily.com/
    "serper_api_key": "",          # Daftar gratis (2500/bulan) di https://serper.dev/
    "internal_thought_enabled": True,
    "token_budget": {
        "total_ctx": 8192,
        "response_reserved": 512,
        "system_identity": 350,
        "memory_budget": 600,
        # sisa = conversation history
    }
}

def load_config() -> dict:
    if not CONFIG_PATH.exists():
        save_config(DEFAULT_CONFIG)
        return DEFAULT_CONFIG.copy()
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Merge dengan default agar key baru selalu ada
        merged = DEFAULT_CONFIG.copy()
        merged.update(data)
        return merged
    except (json.JSONDecodeError, ValueError):
        save_config(DEFAULT_CONFIG)
        return DEFAULT_CONFIG.copy()

def save_config(cfg: dict):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

def setup_wizard(cfg: dict) -> dict:
    """
    Wizard setup hanya dijalankan saat pertama kali atau user minta reset.
    Menyimpan hasil ke config.json agar tidak ditanya lagi.
    """
    print("\n" + "="*50)
    print("  ASTA — Setup Awal (hanya sekali)")
    print("="*50)

    # Model
    print("\nPilih model:")
    print("  1. Sailor2 3B (lebih ringan)")
    print("  2. Sailor2 8B (lebih pintar)")
    choice = input("Pilihan (default = 1): ").strip() or "1"
    cfg["model_choice"] = choice if choice in ["1", "2"] else "1"

    # Device
    print("\nPilih device:")
    print("  1. CPU")
    print("  2. GPU CUDA")
    dev = input("Pilihan (default = 1): ").strip()
    cfg["device"] = "gpu" if dev == "2" else "cpu"

    # LoRA
    use_lora = input("\nGunakan LoRA adapter? (y/n, default = n): ").strip().lower()
    cfg["use_lora"] = use_lora == "y"
    if cfg["use_lora"]:
        try:
            lg = input("Layer LoRA di GPU? (default = 0): ").strip()
            cfg["lora_n_gpu_layers"] = int(lg) if lg else 0
        except ValueError:
            cfg["lora_n_gpu_layers"] = 0

    # Web search
    ws = input("\nAktifkan web search? (y/n, default = y): ").strip().lower()
    cfg["web_search_enabled"] = ws != "n"
    if cfg["web_search_enabled"]:
        print("  Tavily API key (1000 query/bulan gratis, REKOMENDASI):")
        print("  Daftar di https://app.tavily.com/")
        tavily_key = input("  Tavily API key (kosong = skip): ").strip()
        cfg["tavily_api_key"] = tavily_key
        if not tavily_key:
            print("  Serper API key (2500 query/bulan gratis, alternatif):")
            print("  Daftar di https://serper.dev/")
            serper_key = input("  Serper API key (kosong = pakai DDG+Wikipedia saja): ").strip()
            cfg["serper_api_key"] = serper_key

    # Internal thought
    it = input("Aktifkan internal thought? (y/n, default = y): ").strip().lower()
    cfg["internal_thought_enabled"] = it != "n"

    save_config(cfg)
    print("\n✓ Konfigurasi disimpan ke config.json")
    print("  Untuk reset, hapus file config.json atau jalankan: python core.py --setup\n")
    return cfg
