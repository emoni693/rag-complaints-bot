# run_task1.py (updated version)
import os
import yaml
import pandas as pd
import re

def clean_text(text):
    """Simple text cleaning"""
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

def main():
    print("=== RAG Complaints Bot - Task 1: Data Preprocessing ===")
    
    # Check if config exists
    config_path = "configs/config.yaml"
    if not os.path.exists(config_path):
        print(f"❌ Config file not found at {config_path}")
        print("Please create configs/config.yaml first")
        return
    
    # Load config
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        print("✅ Config loaded successfully")
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        return
    
    raw_dir = cfg["paths"]["raw_dir"]
    processed_dir = cfg["paths"]["processed_dir"]
    
    # Check raw directory
    if not os.path.exists(raw_dir):
        print(f"❌ Raw data directory not found: {raw_dir}")
        print("Please create the data/raw/ folder and place your CSV file there")
        return
    
    # Look for CSV files
    csv_files = [f for f in os.listdir(raw_dir) if f.endswith('.csv')]
    
    if not csv_files:
        print(f"❌ No CSV files found in {raw_dir}")
        print("Please place your CFPB complaints CSV file in the data/raw/ folder")
        return
    
    # Use the first CSV file found
    raw_csv = os.path.join(raw_dir, csv_files[0])
    print(f"📁 Found CSV file: {csv_files[0]}")
    
    # Load data
    try:
        print("📊 Loading CSV data...")
        df = pd.read_csv(raw_csv, dtype=str)
        print(f"✅ Loaded {len(df)} rows from CSV")
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return
    
    # Show available columns
    print(f"\n�� Available columns: {list(df.columns)}")
    
    # Map actual column names to expected names
    column_mapping = {
        'Product': 'product',
        'Issue': 'issue', 
        'Consumer complaint narrative': 'consumer_complaint_narrative',
        'Date received': 'date_received',
        'Company': 'company',
        'Complaint ID': 'complaint_id'
    }
    
    # Check which columns we have and which are missing
    available_columns = list(df.columns)
    missing_columns = []
    
    for expected_col, actual_col in column_mapping.items():
        if expected_col in available_columns:
            print(f"✅ Found column: {expected_col} -> {actual_col}")
        else:
            missing_columns.append(expected_col)
            print(f"❌ Missing column: {expected_col}")
    
    if missing_columns:
        print(f"\n⚠️  Warning: Missing columns: {missing_columns}")
        print("The script will continue but may not work properly")
    
    # Rename columns to match expected names
    print("\n🔄 Renaming columns...")
    df = df.rename(columns=column_mapping)
    
    # Filter products
    print("\n�� Filtering products...")
    df = df.copy()
    df["product"] = df["product"].str.lower().str.strip()
    allowed = {p.lower() for p in cfg["products"]}
    
    print(f"Allowed products: {list(allowed)}")
    print(f"Available products in data: {list(df['product'].unique())}")
    
    # Show product distribution before filtering
    print("\n📊 Product distribution before filtering:")
    product_counts = df['product'].value_counts()
    for product, count in product_counts.head(10).items():
        print(f"  {product}: {count}")
    
    df = df[df["product"].isin(allowed)]
    print(f"\n✅ After product filtering: {len(df)} rows")
    
    # Clean narratives
    print("\n�� Cleaning complaint narratives...")
    df["consumer_complaint_narrative"] = df["consumer_complaint_narrative"].map(clean_text)
    
    # Remove empty narratives
    df = df[df["consumer_complaint_narrative"].str.len() > 0]
    print(f"✅ After removing empty narratives: {len(df)} rows")
    
    # Ensure complaint_id is string
    df["complaint_id"] = df["complaint_id"].astype(str)
    df = df.reset_index(drop=True)
    
    # Create processed directory
    os.makedirs(processed_dir, exist_ok=True)
    
    # Save processed data
    out_csv = cfg["paths"]["filtered_csv"]
    df.to_csv(out_csv, index=False)
    
    print(f"\n�� Successfully wrote {out_csv} with {len(df)} rows.")
    
    # Show analysis
    print("\n" + "="*50)
    print("📊 DATA ANALYSIS RESULTS")
    print("="*50)
    
    print(f"Total complaints: {len(df)}")
    print(f"Products: {', '.join(df['product'].unique())}")
    
    print("\n�� PRODUCT COUNTS:")
    for product in df['product'].unique():
        count = len(df[df['product'] == product])
        print(f"  {product}: {count} complaints")
    
    print("\n📏 TEXT LENGTH ANALYSIS:")
    lengths = df['consumer_complaint_narrative'].str.len()
    print(f"  Shortest: {lengths.min()} characters")
    print(f"  Longest: {lengths.max()} characters")
    print(f"  Average: {lengths.mean():.0f} characters")
    
    print("\n📝 SAMPLE COMPLAINTS:")
    for product in df['product'].unique()[:2]:
        sample = df[df['product'] == product].iloc[0]
        print(f"\n  {product.upper()}:")
        print(f"  Text: {sample['consumer_complaint_narrative'][:100]}...")
    
    print("\n✅ Task 1 completed successfully!")

if __name__ == "__main__":
    main()