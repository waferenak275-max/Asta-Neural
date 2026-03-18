import json
from pathlib import Path

CONFIG_PATH = Path("config.json")

DEFAULT_CONFIG = {
    "model_choice": "2",
    "device": "cpu",
    "use_lora": False,
    "lora_n_gpu_layers": 0,
    "memory_mode": "hybrid",
    "web_search_enabled": True,
    "n_batch": 1024,
    "tavily_api_key": "",
    "serper_api_key": "",
    "internal_thought_enabled": True,
    "internal_thought_combined_steps": False,
    "use_dynamic_prompt": True,
    "thought_n_ctx": 1024,
    "token_budget": {
        "total_ctx": 8192,
        "response_reserved": 512,
        "system_identity": 350,
        "memory_budget": 600,
    }
}

def load_config() -> dict:
    if not CONFIG_PATH.exists():
        save_config(DEFAULT_CONFIG)
        return DEFAULT_CONFIG.copy()
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
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
    print("\n" + "="*50)
    print("  ASTA — Setup Awal (hanya sekali)")
    print("="*50)

    print("\nPilih model response:")
    print("  1. Sailor2 3B (lebih ringan)")
    print("  2. Sailor2 8B (lebih pintar) [default]")
    choice = input("Pilihan (default = 2): ").strip() or "2"
    cfg["model_choice"] = choice if choice in ["1", "2"] else "2"

    print("\nPilih device:")
    print("  1. CPU [default]")
    print("  2. GPU CUDA")
    dev = input("Pilihan (default = 1): ").strip()
    cfg["device"] = "gpu" if dev == "2" else "cpu"

    use_lora = input("\nGunakan LoRA adapter? (y/n, default = n): ").strip().lower()
    cfg["use_lora"] = use_lora == "y"
    if cfg["use_lora"]:
        try:
            lg = input("Layer LoRA di GPU? (default = 0): ").strip()
            cfg["lora_n_gpu_layers"] = int(lg) if lg else 0
        except ValueError:
            cfg["lora_n_gpu_layers"] = 0

    ws = input("\nAktifkan web search? (y/n, default = y): ").strip().lower()
    cfg["web_search_enabled"] = ws != "n"
    if cfg["web_search_enabled"]:
        print("  Tavily API key (1000 query/bulan gratis, REKOMENDASI):")
        tavily_key = input("  Tavily API key (kosong = skip): ").strip()
        cfg["tavily_api_key"] = tavily_key
        if not tavily_key:
            print("  Serper API key (2500 query/bulan gratis, alternatif):")
            serper_key = input("  Serper API key (kosong = pakai DDG+Wikipedia saja): ").strip()
            cfg["serper_api_key"] = serper_key

    it = input("Aktifkan internal thought? (y/n, default = y): ").strip().lower()
    cfg["internal_thought_enabled"] = it != "n"

    if cfg["internal_thought_enabled"]:
        combined = input("Gunakan mode combined thought 4-step jadi 1x inferensi? (y/n, default = n): ").strip().lower()
        cfg["internal_thought_combined_steps"] = combined == "y"

    if cfg["internal_thought_enabled"] and cfg["model_choice"] == "2":
        print("\n[Info] Thought model akan menggunakan Sailor2 3B secara terpisah.")
        print("  n_ctx untuk thought (default = 2048, lebih kecil = hemat RAM):")
        try:
            nc = input("  thought_n_ctx (default = 2048): ").strip()
            cfg["thought_n_ctx"] = int(nc) if nc else 2048
        except ValueError:
            cfg["thought_n_ctx"] = 2048

    save_config(cfg)
    print("\n✓ Konfigurasi disimpan ke config.json")
    print("  Untuk reset: python core.py --setup\n")
    return cfg
