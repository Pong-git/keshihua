from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "MC1" / "V1" / "MC1.json"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

SUSPECT_ENTITIES = [
    "Mar de la Vida OJSC",
    979893388,
    "Oceanfront Oasis Inc Carriers",
    8327,
]

NODE_TYPE_COLORS = {
    "company": "#1f77b4",
    "person": "#ff7f0e",
    "organization": "#2ca02c",
    "political_organization": "#9467bd",
    "location": "#8c564b",
    "vessel": "#17becf",
    "event": "#d62728",
    "movement": "#7f7f7f",
    "unknown": "#9ca3af",
}

EDGE_TYPE_COLORS = {
    "ownership": "#d62728",
    "membership": "#2ca02c",
    "partnership": "#1f77b4",
    "family_relationship": "#ff7f0e",
    "unknown": "#9ca3af",
}

ILLEGAL_FISHING_KEYWORDS = [
    "illegal",
    "unregulated",
    "fishing",
    "fish",
    "shrimp",
    "vessel",
    "maritime",
    "ocean",
    "sea",
    "catch",
    "harbor",
]
