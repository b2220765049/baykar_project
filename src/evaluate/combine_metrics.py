import pandas as pd
import os

def combine_metrics():
    # Define file paths
    base_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'evaluate')
    manual_metrics_path = os.path.join(base_path, 'manuel_metricsV3.csv')
    rag_metrics_path = os.path.join(base_path, 'rag_evaluation_reportV3.csv')
    output_path = os.path.join(base_path, 'combined_evaluation_metrics.csv')

    # Read both CSV files
    manual_df = pd.read_csv(manual_metrics_path)
    rag_df = pd.read_csv(rag_metrics_path)

    # Merge the dataframes on the query field
    combined_df = pd.merge(
        manual_df,
        rag_df,
        left_on='query',
        right_on='Query',
        how='outer'  # Keep all rows from both dataframes
    )

    # Drop duplicate columns (capitalized versions)
    columns_to_drop = ['Query', 'Generated_Answer', 'Expected_Answer']
    combined_df = combined_df.drop(columns=columns_to_drop, errors='ignore')

    # Save the combined dataframe
    combined_df.to_csv(output_path, index=False)
    print(f"Combined metrics saved to: {output_path}")
    print(f"Total number of rows: {len(combined_df)}")
    print("\nColumns in combined dataset:")
    for col in combined_df.columns:
        print(f"- {col}")

if __name__ == '__main__':
    combine_metrics()