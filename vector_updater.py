import time
from datetime import datetime
from main import (  # Reuse core logic from original
    poll_drive_folder, process_file,
    load_processed_files, save_processed_files
)

def update_files():
    print(f"\n=== Update {datetime.now().isoformat()} ===")
    processed = load_processed_files()
    
    try:
        new_files = [
            f for f in poll_drive_folder()
            if f['id'] not in processed
        ]
        
        for file in new_files:
            process_file(file)
            processed.append(file['id'])
            save_processed_files(processed)
            
        print(f"Processed {len(new_files)} new files")
        
    except Exception as e:
        print(f"Update failed: {str(e)}")

if __name__ == "__main__":
    while True:
        update_files()
        time.sleep(3600)  # 1 hour