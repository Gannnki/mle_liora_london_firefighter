import os
import pandas as pd

# 1. Define relative paths based on your architecture
# '../' moves one directory up to locate your heavy source file
input_csv_path = "../data/dataset_with_filtered_distance_speed.csv"  
output_dir = "data_streamlit"
output_csv_path = os.path.join(output_dir, "lfb_map_sample.csv")

print("🔄 Loading raw dataset from one directory above...")

try:
    # Load only the requested coordinate columns to minimize memory usage
    df_raw = pd.read_csv(input_csv_path, usecols=["Latitude", "Longitude"])
    
    # Drop rows containing missing values (NaNs)
    df_map = df_raw.dropna().copy()
    
    # Map column headers to lowercase 'lat' and 'lon' as strictly required by st.map
    df_map.columns = ["lat", "lon"]
    
    # Determine the safest sample boundary (maximum 5,000 for fluid rendering)
    sample_size = min(5000, len(df_map))
    
    print(f"📊 Total valid coordinate rows discovered: {len(df_map)}")
    print(f"🎲 Drawing a random sample of {sample_size} rows to prevent browser lagging...")
    
    # Extract a reproducible random spatial sample
    df_sample = df_map.sample(n=sample_size, random_state=42)
    
    # Guarantee that the target assets directory exists on disk
    #os.makedirs(output_dir, exist_ok=True)
    
    # Save the lightweight dataframe sample as a flat deployment asset
    df_sample.to_csv(output_csv_path, index=False)
    
    print(f"✅ Success! True spatial telemetry saved to: {output_csv_path}")
    print("🚀 You can now safely delete this script and refresh your Streamlit browser tab.")

except FileNotFoundError:
    print(f"❌ FileNotFoundError: Could not locate the file '{input_csv_path}' one directory above.")
    print("💡 Action: Please double-check your raw CSV filename and update line 6 of this script!")
except Exception as e:
    print(f"❌ Unexpected Runtime Error: {e}")