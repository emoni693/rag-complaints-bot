# scripts/make_dataset.py
import os
import sys
import yaml

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.data.loader import load_and_prepare

def analyze_data_simple(df):
    """Simple text-based analysis without charts"""
    print("=== COMPLAINT ANALYSIS ===")
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
    for product in df['product'].unique()[:2]:  # Show first 2 products
        sample = df[df['product'] == product].iloc[0]
        print(f"\n{product.upper()}:")
        print(f"Text: {sample['consumer_complaint_narrative'][:100]}...")

if __name__ == "__main__":
    with open("configs/config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    
    raw_dir = cfg["paths"]["raw_dir"]
    processed_dir = cfg["paths"]["processed_dir"]
    os.makedirs(processed_dir, exist_ok=True)
    
    # Replace filename below with your raw CFPB CSV filename
    raw_csv = os.path.join(raw_dir, "cfpb_complaints.csv")
    df = load_and_prepare(raw_csv, cfg["products"])
    
    out_csv = cfg["paths"]["filtered_csv"]
    df.to_csv(out_csv, index=False)
    
    print(f"\n✅ Wrote {out_csv} with {len(df)} rows.")
    
    # Show analysis
    analyze_data_simple(df)