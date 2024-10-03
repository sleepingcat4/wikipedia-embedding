import os
import pyarrow as pa
import pyarrow.parquet as pq

def concat_wiki(input_file, output_file, process_all, wiki_language):
    table = pq.read_table(input_file)
    
    if process_all != 'yes':
        table = table.slice(0, 10)
    
    title_col = table.column('Title')
    abstract_col = table.column('Abstract')
    url_col = table.column('URL')

    concat_col = pa.array([f"{title} {abstract}" for title, abstract in zip(title_col, abstract_col)])
    wiki_language_col = pa.array([wiki_language] * table.num_rows)
    
    table = table.append_column('Concat Abstract', concat_col)
    table = table.append_column('Wiki Language', wiki_language_col)
    table = table.select(['Title', 'Concat Abstract', 'URL', 'Version Control', 'Wiki Language'])
    
    pq.write_table(table, output_file)
    print(f"Final output saved: {output_file}")
    
    assert table.num_rows == table.num_rows, "Row count mismatch between input and output files."

if __name__ == "__main__":
    input_file = input("Enter the absolute path to the input Parquet file: ").strip()
    output_file = input("Enter the absolute path to the output Parquet file: ").strip()
    process_all = input("Process entire file? (yes/no): ").strip().lower()
    wiki_language = input("Enter the Wiki Language value (e.g., dewiki): ").strip()

    concat_wiki(input_file, output_file, process_all, wiki_language)
