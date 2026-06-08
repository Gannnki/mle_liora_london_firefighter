import os
import pandas as pd

def export_to_csv(export_objects: dict, target_col: str, output_dir: str = "data_splits"):
    os.makedirs(output_dir, exist_ok=True)

    summary_rows = []

    for name, obj in export_objects.items():
        if obj is None:
            continue

        if isinstance(obj, pd.Series):
            col_name = f"{target_col}_log" if "log" in name else target_col
            obj = obj.to_frame(name=col_name)

        file_path = os.path.join(output_dir, f"{name}.csv")
        obj.to_csv(file_path, index=False)

        summary_rows.append({
            "file_name": f"{name}.csv",
            "rows": obj.shape[0],
            "cols": obj.shape[1],
            "path": file_path,
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(
        os.path.join(output_dir, "split_export_summary.csv"),
        index=False
    )

    print(f"Exported {len(summary_rows)} files to: {output_dir}")