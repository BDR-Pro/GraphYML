"""
Utility to convert CSV data to YAML format.
Handles parsing and transformation of various field types.
"""
import csv
import yaml
from ast import literal_eval
from pathlib import Path


def parse_list_field(value):
    """
    Convert stringified list or comma-separated string to Python list.
    
    Args:
        value (str): String to parse
        
    Returns:
        list: Parsed list
    """
    if not value:
        return []
    
    try:
        if value.startswith("["):
            # Try to parse as a literal list
            return literal_eval(value)
        else:
            # Treat as comma-separated values
            return [v.strip() for v in value.split(",")]
    except (ValueError, SyntaxError):
        # If parsing fails, return as a single-item list
        return [value]


def convert_csv_to_yaml(csv_path, output_dir, id_field="imdb_id"):
    """
    Convert CSV file to individual YAML files.
    
    Args:
        csv_path (str): Path to CSV file
        output_dir (str): Directory to save YAML files
        id_field (str): Field to use as ID
        
    Returns:
        tuple: (count_success, count_error, errors)
    """
    # Ensure output directory exists
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    count_success = 0
    count_error = 0
    errors = []
    
    try:
        with open(csv_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            
            for row in reader:
                try:
                    # Determine ID for the file
                    movie_id = (
                        row.get(id_field) or 
                        row.get("id") or 
                        row.get("original_title", "unknown")
                    )
                    
                    # Clean movie_id for filename
                    safe_id = "".join(c if c.isalnum() else "_" for c in movie_id)
                    
                    # Create node dictionary
                    node = {
                        "id": row.get(id_field),
                        "title": row.get("original_title"),
                        "tagline": row.get("tagline"),
                        "director": row.get("director"),
                        "cast": parse_list_field(row.get("cast")),
                        "genres": parse_list_field(row.get("genres")),
                        "keywords": parse_list_field(row.get("keywords")),
                        "tags": parse_list_field(row.get("keywords")),  # Duplicate as tags for compatibility
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
                    
                    # Remove None values
                    node = {k: v for k, v in node.items() if v is not None}
                    
                    # Write to YAML file
                    output_file = output_path / f"{safe_id}.yaml"
                    with open(output_file, "w", encoding="utf-8") as f:
                        yaml.dump(node, f, sort_keys=False)
                    
                    count_success += 1
                    
                except Exception as e:
                    count_error += 1
                    errors.append((movie_id, str(e)))
    
    except Exception as e:
        errors.append(("file_error", str(e)))
    
    return count_success, count_error, errors


def main():
    """Command-line entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert CSV to YAML files")
    parser.add_argument("csv_path", help="Path to CSV file")
    parser.add_argument("output_dir", help="Directory to save YAML files")
    parser.add_argument("--id-field", default="imdb_id", help="Field to use as ID")
    
    args = parser.parse_args()
    
    print(f"Converting {args.csv_path} to YAML files in {args.output_dir}...")
    success, error, errors = convert_csv_to_yaml(
        args.csv_path, 
        args.output_dir,
        args.id_field
    )
    
    print(f"Conversion complete: {success} successful, {error} failed")
    
    if errors:
        print(f"Errors encountered:")
        for item, error in errors[:10]:  # Show first 10 errors
            print(f"- {item}: {error}")
        
        if len(errors) > 10:
            print(f"... and {len(errors) - 10} more errors")


if __name__ == "__main__":
    main()

