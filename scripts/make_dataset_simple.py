# scripts/make_dataset_simple.py
import os
import yaml
import pandas as pd
import re

def clean_text(text):
    """Simple text cleaning without complex regex"""
    if text is None or pd.isna(text):
        return ""
    
    # Convert to string and normalize whitespace
    normalized = re.sub(r"\s+", " ", str(text)).strip()
    
    # Convert to lowercase
    normalized = normalized.lower()
    
    # Simple text replacements
    boilerplate_phrases = [
        "thank you for your help",
        "please contact me", 
        "i have attached",
        "please help",
        "thank you",
        "best regards",
        "sincerely",
    ]
    
    for phrase in boilerplate_phrases:
        normalized = normalized.replace(phrase, "")
    
    # Clean up extra whitespace
    normalized = re.sub(r"\s+", " ", normalized).strip()
    
    return normalized

def load_and_prepare(csv_path, allowed_products):
    """Load and prepare the complaint data"""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing dataset at {csv_path}")
    
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path, dtype=str)
    
    # Check required columns
    required_columns = ["product", "issue", "consumer_complaint_narrative", "date_received", "company", "complaint_id"]
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        print(f"Warning: Missing columns: {missing}")
        print(f"Available columns: {list(df.columns)}")
        return None
    
    print(f"Original dataset: {len(df)} rows")
    
    # Filter products
    df = df.copy()
    df["product"] = df["product"].str.lower().str.strip()
    allowed = {p.lower() for p in allowed_products}
    df = df[df["product"].isin(allowed)]
    print(f"After product filtering: {len(df)} rows")
    
    # Clean narratives
    print("Cleaning complaint narratives...")
    df["consumer_complaint_narrative"] = df["consumer_complaint_narrative"].map(clean_text)
    
    # Remove empty narratives
    df = df[df["consumer_complaint_narrative"].str.len() > 0]
    print(f"After removing empty narratives: {len(df)} rows")
    
    # Ensure complaint_id is string
    df["complaint_id"] = df["complaint_id"].astype(str)
    df = df.reset_index(drop=True)
    
    return df

def analyze_data_simple(df):
    """Simple text-based analysis"""
    print("\n=== COMPLAINT ANALYSIS ===")
    print(f"Total complaints: {len(df)}")
    print(f"Products: {', '.join(df['product'].unique())}")
    
    print("\n=== PRODUCT COUNTS ===")
    for product in df['product'].unique():
        count = len(df[df['product'] == product])
        print(f"{product}: {count} complaints")
    
    print("\n=== TEXT LENGTH ANALYSIS ===")
    lengths = df['consumer_complaint_narrative'].str.len()
    print(f"Shortest: {lengths.min()} characters")
    print(f"Longest: {lengths.max()} characters")
    print(f"Average: {lengths.mean():.0f} characters")
    
    print("\n=== SAMPLE COMPLAINTS ===")
    for product in df['product'].unique()[:2]:
        sample = df[df['product'] == product].iloc[0]
        print(f"\n{product.upper()}:")
        print(f"Text: {sample['consumer_complaint_narrative'][:100]}...")

def main():
    # Load config
    config_path = "configs/config.yaml"
    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        print("Please create configs/config.yaml first")
        return
    
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    
    raw_dir = cfg["paths"]["raw_dir"]
    processed_dir = cfg["paths"]["processed_dir"]
    
    # Create processed directory
    os.makedirs(processed_dir, exist_ok=True)
    
    # Look for CSV files in raw directory
    csv_files = [f for f in os.listdir(raw_dir) if f.endswith('.csv')]
    
    if not csv_files:
        print(f"Error: No CSV files found in {raw_dir}")
        print("Please place your CFPB complaints CSV file in the data/raw/ folder")
        return
    
    # Use the first CSV file found
    raw_csv = os.path.join(raw_dir, csv_files[0])
    print(f"Using CSV file: {csv_files[0]}")
    
    # Load and prepare data
    df = load_and_prepare(raw_csv, cfg["products"])
    
    if df is None:
        print("Failed to load data. Please check your CSV file.")
        return
    
    # Save processed data
    out_csv = cfg["paths"]["filtered_csv"]
    df.to_csv(out_csv, index=False)
    
    print(f"\n✅ Successfully wrote {out_csv} with {len(df)} rows.")
    
    # Show analysis
    analyze_data_simple(df)

if __name__ == "__main__":
    main()