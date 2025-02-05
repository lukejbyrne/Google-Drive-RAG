import time
from datetime import datetime
from main import (  # Reuse core logic from original
    poll_drive_folder, process_file,
    load_processed_files, save_processed_files,
    save_processed_files, delete_vectors, console,  # Reuse core logic from original
)

def update_files():
    print(f"\n=== Update {datetime.now().isoformat()} ===")
    processed = load_processed_files()
    
    try:
        # Get current files with empty list fallback
        current_files = poll_drive_folder() or []
        current_ids = {f['id'] for f in current_files}
        
        # Handle deletions
        for file_id in list(processed.keys()):
            if file_id not in current_ids:
                console.print(f"Deleting vectors for removed file: {file_id}")
                if delete_vectors(file_id):
                    del processed[file_id]
                    save_processed_files(processed)
        
        # Process new/updated files
        for file in current_files:
            existing = processed.get(file['id'])
            if not existing or file['modifiedTime'] > existing['modified']:
                process_file(file)
                
    except Exception as e:
        console.print(f"Update failed: {str(e)}")
        raise  # Preserve stack trace

if __name__ == "__main__":
    while True:
        update_files()
        time.sleep(3600)  # 1 hour