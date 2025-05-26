import csv
import yaml
from ast import literal_eval
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "tmdb-movies.csv"
YML_DIR = BASE_DIR / "converted_yml"
YML_DIR.mkdir(exist_ok=True)

def parse_list_field(value: str):
    """Convert stringified list or comma-separated string to Python list."""
    if not value:
        return []
    try:
        return literal_eval(value) if value.startswith("[") else [v.strip() for v in value.split(",")]
    except (ValueError, SyntaxError):
        return [value]

# Process CSV and write individual YAML files
with CSV_PATH.open(newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        movie_id = row.get("imdb_id") or row.get("id") or row.get("original_title", "unknown")

        node = {
            "id": row.get("imdb_id"),
            "title": row.get("original_title"),
            "tagline": row.get("tagline"),
            "director": row.get("director"),
            "cast": parse_list_field(row.get("cast")),
            "genres": parse_list_field(row.get("genres")),
            "keywords": parse_list_field(row.get("keywords")),
            "overview": row.get("overview"),
            "runtime": float(row.get("runtime") or 0),
            "release_date": row.get("release_date"),
            "vote_count": int(row.get("vote_count") or 0),
            "vote_average": float(row.get("vote_average") or 0),
            "budget": float(row.get("budget") or 0),
            "revenue": float(row.get("revenue") or 0),
            "budget_adj": float(row.get("budget_adj") or 0),
            "revenue_adj": float(row.get("revenue_adj") or 0),
            "popularity": float(row.get("popularity") or 0),
            "production_companies": parse_list_field(row.get("production_companies")),
            "year": int(row.get("release_year") or 0),
            "links": []
        }

        output_file = YML_DIR / f"{movie_id}.yaml"
        with output_file.open("w", encoding="utf-8") as f:
            yaml.dump(node, f, sort_keys=False)
