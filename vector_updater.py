
import sys
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

def wait_or_pull(interval=3600):
    """Waits for the interval, but allows manual execution with 'pull' or quitting with 'q'."""
    start_time = time.time()
    while time.time() - start_time < interval:
        user_input = input("Type 'pull' to run immediately or 'q' to quit: ").strip().lower()
        if user_input == "pull":
            return  # Exit wait early and trigger update immediately
        elif user_input == "q":
            print("Exiting program...")
            sys.exit(0)  # Exit gracefully
        time.sleep(1)  # Small sleep to avoid constant CPU usage

if __name__ == "__main__":
    while True:
        update_files()
        wait_or_pull()  # Wait for an hour, but allow manual trigger or quit