"""Tek komut pipeline: data/ klasorundeki PDF'leri tara, degisenleri reindex et.

Kullanim:
    python pipeline.py            # incremental (sadece yeni/degisen)
    python pipeline.py --force    # tam rebuild
"""
import sys
import json

from core.vector_engine import build_or_update


def main():
    force = "--force" in sys.argv
    print(f"[pipeline] {'FORCE REBUILD' if force else 'INCREMENTAL UPDATE'}")
    report = build_or_update(force=force)
    print("\n[pipeline] Tamamlandi:")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
