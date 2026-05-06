"""List installed face-swap presets (slug + display name)."""
from .presets import list_presets_detailed, TARGETS_DIR


def main():
    entries = list_presets_detailed()
    print(f"Targets dir: {TARGETS_DIR}")
    print(f"{len(entries)} preset(s):\n")
    for e in entries:
        print(f"  {e['id']:10s} {e['name']}")


if __name__ == "__main__":
    main()
