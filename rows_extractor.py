import pandas as pd
import os

def process_large_tsv(input_file, column_name='X', output_file='filtered_output.csv', chunksize=10000):
    """
    Process a large TSV file in chunks and create a CSV with rows where column_name is not empty/null.
    
    Parameters:
    -----------
    input_file : str
        Path to input TSV file
    column_name : str
        Name of the column to check for non-empty values (default: 'X')
    output_file : str
        Path to output CSV file (default: 'filtered_output.csv')
    chunksize : int
        Number of rows to process at a time (default: 10000)
    """
    
    first_chunk = True
    total_rows_processed = 0
    total_rows_kept = 0
    
    print(f"Processing {input_file}...")
    print(f"Filtering rows where column '{column_name}' is not empty/null")
    print(f"Chunk size: {chunksize:,} rows")
    print("-" * 50)
    
    try:
        # Process file in chunks
        for i, chunk in enumerate(pd.read_csv(input_file, sep='\t', chunksize=chunksize, 
                                              low_memory=False), 1):
            
            total_rows_processed += len(chunk)
            
            # Filter rows where column 'X' is not empty/null
            # This filters out: NaN, None, empty strings, and strings with only whitespace
            filtered_chunk = chunk[
                chunk[column_name].notna() & 
                (chunk[column_name].astype(str).str.strip() != '')
            ]
            
            rows_kept = len(filtered_chunk)
            total_rows_kept += rows_kept
            
            # Write to CSV
            if not filtered_chunk.empty:
                mode = 'w' if first_chunk else 'a'
                header = first_chunk
                filtered_chunk.to_csv(output_file, mode=mode, header=header, index=False)
                first_chunk = False
            
            # Progress update
            if i % 10 == 0:
                print(f"Chunk {i}: Processed {total_rows_processed:,} rows, "
                      f"kept {total_rows_kept:,} rows ({total_rows_kept/total_rows_processed*100:.2f}%)")
        
        print("-" * 50)
        print(f"✓ Processing complete!")
        print(f"Total rows processed: {total_rows_processed:,}")
        print(f"Total rows kept: {total_rows_kept:,}")
        print(f"Output saved to: {output_file}")
        print(f"File size: {os.path.getsize(output_file) / (1024*1024):.2f} MB")
        
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found!")
    except KeyError:
        print(f"Error: Column '{column_name}' not found in the TSV file!")
        print("Available columns might be different. Check your column name.")
    except Exception as e:
        print(f"Error occurred: {str(e)}")


def create_multiple_datasets(input_file, column_name='X', output_dir='output_datasets', 
                             chunksize=10000, max_rows_per_file=100000):
    """
    Create multiple CSV files, splitting filtered data into smaller datasets.
    
    Parameters:
    -----------
    input_file : str
        Path to input TSV file
    column_name : str
        Name of the column to check for non-empty values
    output_dir : str
        Directory to save output CSV files
    chunksize : int
        Number of rows to read at a time
    max_rows_per_file : int
        Maximum rows per output CSV file
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    file_counter = 1
    rows_in_current_file = 0
    total_rows_processed = 0
    total_rows_kept = 0
    first_chunk_in_file = True
    current_output_file = os.path.join(output_dir, f'dataset_{file_counter}.csv')
    
    print(f"Processing {input_file}...")
    print(f"Creating multiple datasets in '{output_dir}/'")
    print(f"Max rows per file: {max_rows_per_file:,}")
    print("-" * 50)
    
    try:
        for i, chunk in enumerate(pd.read_csv(input_file, sep='\t', chunksize=chunksize, 
                                              low_memory=False), 1):
            
            total_rows_processed += len(chunk)
            
            # Filter rows
            filtered_chunk = chunk[
                chunk[column_name].notna() & 
                (chunk[column_name].astype(str).str.strip() != '')
            ]
            
            if filtered_chunk.empty:
                continue
            
            # Split chunk if it would exceed max_rows_per_file
            remaining_rows = filtered_chunk
            
            while not remaining_rows.empty:
                space_in_file = max_rows_per_file - rows_in_current_file
                rows_to_write = remaining_rows.iloc[:space_in_file]
                
                # Write to current file
                mode = 'w' if first_chunk_in_file else 'a'
                header = first_chunk_in_file
                rows_to_write.to_csv(current_output_file, mode=mode, header=header, index=False)
                
                rows_in_current_file += len(rows_to_write)
                total_rows_kept += len(rows_to_write)
                first_chunk_in_file = False
                
                # Check if we need a new file
                if rows_in_current_file >= max_rows_per_file:
                    print(f"✓ Created {current_output_file} ({rows_in_current_file:,} rows)")
                    file_counter += 1
                    current_output_file = os.path.join(output_dir, f'dataset_{file_counter}.csv')
                    rows_in_current_file = 0
                    first_chunk_in_file = True
                
                # Move to remaining rows
                remaining_rows = remaining_rows.iloc[space_in_file:]
            
            # Progress update
            if i % 10 == 0:
                print(f"Processed {total_rows_processed:,} rows, kept {total_rows_kept:,} rows")
        
        # Final file info
        if rows_in_current_file > 0:
            print(f"✓ Created {current_output_file} ({rows_in_current_file:,} rows)")
        
        print("-" * 50)
        print(f"✓ Processing complete!")
        print(f"Total rows processed: {total_rows_processed:,}")
        print(f"Total rows kept: {total_rows_kept:,}")
        print(f"Created {file_counter} dataset file(s)")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")


if __name__ == "__main__":
    # Example usage - choose one:
    
    # Option 1: Create single CSV file
    process_large_tsv(
        input_file='C:\\Users\\iyed\\Desktop\\BindingDB_All_202509_tsv\\Protein-Ligand-Binding-affinity-prediction-\\BindingDB_All.tsv',
        column_name='IC50 (nM)',
        output_file='IC50 (nM).csv',
        chunksize=300_000  # Adjust based on your memory
    )
    
    # Option 2: Create multiple CSV files (uncomment to use)
    # create_multiple_datasets(
    #     input_file='your_file.tsv',
    #     column_name='X',
    #     output_dir='output_datasets',
    #     chunksize=10000,
    #     max_rows_per_file=100000
    # )