import argparse
from pathlib import Path

from modeling_dl import TemporalSequenceBuilder


DEFAULT_INPUT_PATH = "data/dataset_with_filtered_distance_speed.csv"
DEFAULT_OUTPUT_PATH = "data/dataset_with_filtered_distance_dl.csv"
DEFAULT_CONFIG_PATH = "config/pipeline_config.yaml"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate station-level temporal workload features for DL/LSTM "
            "experiments."
        )
    )
    parser.add_argument(
        "--input-path",
        default=DEFAULT_INPUT_PATH,
        help="Input CSV path.",
    )
    parser.add_argument(
        "--output-path",
        default=DEFAULT_OUTPUT_PATH,
        help="Output CSV path.",
    )
    parser.add_argument(
        "--config-path",
        default=DEFAULT_CONFIG_PATH,
        help="Pipeline config path.",
    )
    parser.add_argument(
        "--lookback-hours",
        type=int,
        default=24,
        help="Number of previous station-hours used for temporal features.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional row limit for quick smoke tests.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Building DL temporal feature dataset...")
    print(f"Input: {args.input_path}")
    print(f"Output: {output_path}")
    print(f"Lookback hours: {args.lookback_hours}")

    builder = TemporalSequenceBuilder(
        data_path=args.input_path,
        config_path=args.config_path,
        lookback_hours=args.lookback_hours,
        max_rows=args.max_rows,
    )
    feature_df = builder.build_temporal_feature_frame()

    feature_df.to_csv(output_path, index=False)

    print(f"Saved: {output_path}")
    print(f"Rows: {len(feature_df):,}")
    print(f"Columns: {len(feature_df.columns):,}")


if __name__ == "__main__":
    main()
