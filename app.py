import os
import io
import json
import time
import sys
from datetime import datetime

from dotenv import load_dotenv
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# For a nicer console output (optional):
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn

# Pinecone, Google PaLM:
import google.generativeai as palm
from pinecone import Pinecone, ServerlessSpec

console = Console()

# -------------------------------
# 1) Load Environment Variables
# -------------------------------
load_dotenv()

SERVICE_ACCOUNT_FILE = os.environ.get("GOOGLE_DRIVE_SERVICE_ACCOUNT_FILE")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_ENV = os.environ.get("PINECONE_ENV")  # e.g. "us-west-2"
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME")
GOOGLE_GEMINI_API_KEY = os.environ.get("GOOGLE_GEMINI_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GOOGLE_DRIVE_FOLDER_ID = os.environ.get("GOOGLE_DRIVE_FOLDER_ID")

if not all([
    SERVICE_ACCOUNT_FILE, PINECONE_API_KEY, PINECONE_ENV, 
    GOOGLE_GEMINI_API_KEY, GROQ_API_KEY, GOOGLE_DRIVE_FOLDER_ID
]):
    console.print("[red]Error: Missing one or more required environment variables.[/red]")
    sys.exit(1)


# -------------------------------
# 2) Initialize Pinecone
# -------------------------------
pc = Pinecone(api_key=PINECONE_API_KEY)
existing_indexes = [idx["name"] for idx in pc.list_indexes()]

if PINECONE_INDEX_NAME not in existing_indexes:
    console.print(f"[yellow]Index '{PINECONE_INDEX_NAME}' not found. Creating a new one...[/yellow]")
    # Adjust the 'dimension' here to match your chosen embedding model's dimension
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=PINECONE_ENV)
    )

index = pc.Index(PINECONE_INDEX_NAME)

# -------------------------------
# 3) Set Up Google PaLM (Gemini)
# -------------------------------
palm.configure(api_key=GOOGLE_GEMINI_API_KEY)

def get_embedding(text: str) -> list:
    """
    Use Google's PaLM embedding to get an embedding for the given text.
    """
    response = palm.embed_content(
        model="models/text-embedding-004",
        content={"parts": [{"text": text}]}
    )
    if response and "embedding" in response:
        return response["embedding"]
    else:
        console.print(f"[red]Error obtaining embedding: {response}[/red]")
        return None

# -------------------------------
# 4) Google Drive Service Setup
# -------------------------------
SCOPES = ['https://www.googleapis.com/auth/drive']
credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
drive_service = build('drive', 'v3', credentials=credentials)

# -------------------------------
# 5) Utility: Track Processed Files
# -------------------------------
PROCESSED_FILES_PATH = "processed_files.json"

def load_processed_files():
    """
    Returns a dict with { file_id: { modified: str, vectors: [vector_ids], name: str }, ...}
    """
    if os.path.exists(PROCESSED_FILES_PATH):
        with open(PROCESSED_FILES_PATH, "r") as f:
            return json.load(f)
    return {}

def save_processed_files(processed):
    with open(PROCESSED_FILES_PATH, "w") as f:
        json.dump(processed, f, indent=2)

# -------------------------------
# 6) Download a File from Drive
# -------------------------------
def download_file(file_id: str, file_name: str) -> str:
    """
    Download file by ID from Google Drive. Return the file content as a string.
    """
    request = drive_service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    
    console.print(f"[bold green]Downloading file:[/] {file_name}")
    with Progress(
        SpinnerColumn(),
        "[progress.description]{task.description}",
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeElapsedColumn(),
        transient=True,
    ) as progress:
        task = progress.add_task("Downloading...", total=100)
        while not done:
            status, done = downloader.next_chunk()
            if status:
                progress.update(task, completed=int(status.progress() * 100))
    
    fh.seek(0)
    try:
        content = fh.read().decode("utf-8")
    except Exception as e:
        console.print(f"[red]Error decoding file content: {e}[/red]")
        content = ""
    return content

# -------------------------------
# 7) Split Text into Chunks
# -------------------------------
def split_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list:
    """
    Split large text into smaller chunks for embedding.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        if end >= len(text):
            break
        start = end - overlap
    return chunks

# -------------------------------
# 8) Process a Single File
# -------------------------------
def process_file(file: dict):
    """
    Download the file, split it, embed each chunk, upsert into Pinecone, and record in processed_files.json.
    """
    console.rule(f"[bold blue]Processing File: {file['name']}")
    
    # 1. Download
    content = download_file(file["id"], file["name"])
    if not content:
        console.print(f"[red]No content for file {file['name']}[/red]")
        return
    
    # 2. Split
    chunks = split_text(content)
    console.print(f"[green]Split text into {len(chunks)} chunks.[/green]")

    # 3. Embed and Upsert
    vector_ids = []
    with Progress(
        "[progress.description]{task.description}",
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        transient=True,
    ) as progress:
        task = progress.add_task("Embedding & Upserting", total=len(chunks))
        
        for i, chunk in enumerate(chunks):
            embedding = get_embedding(chunk)
            if embedding is None:
                continue
            
            vector_id = f"{file['id']}_{i}"
            vector_ids.append(vector_id)
            
            metadata = {
                "file_id": file["id"],
                "file_name": file["name"],
                "chunk_index": i,
                "text": chunk[:200]  # store a short preview
            }
            
            try:
                index.upsert(
                    vectors=[
                        {
                            "id": vector_id,
                            "values": embedding,
                            "metadata": metadata
                        }
                    ],
                    namespace="default"
                )
            except Exception as e:
                console.print(f"[red]Error upserting vector: {e}[/red]")
            
            time.sleep(0.1)  # slight delay to avoid rate limits
            progress.advance(task)
    
    # 4. Update processed_files.json
    processed = load_processed_files()
    processed[file['id']] = {
        "modified": file["modifiedTime"],
        "vectors": vector_ids,
        "name": file["name"]
    }
    save_processed_files(processed)

    console.print("[bold green]File processed & upserted to Pinecone.[/bold green]\n")

# -------------------------------
# 9) Delete Vectors for a File
# -------------------------------
def delete_vectors(file_id: str):
    """
    Remove existing vectors for a file via known vector IDs or fallback to metadata filter.
    """
    processed = load_processed_files()
    file_data = processed.get(file_id, {})
    
    # Strategy 1: If we have stored vector IDs, try deleting them
    if file_data.get('vectors'):
        try:
            index.delete(ids=file_data['vectors'], namespace="default")
            console.print(f"Deleted {len(file_data['vectors'])} vectors for {file_id}")
            return True
        except Exception as e:
            console.print(f"[red]Vector ID deletion failed: {e}[/red]")

    # Strategy 2: Fallback to metadata filter
    try:
        index.delete(filter={"file_id": {"$eq": file_id}}, namespace="default")
        console.print(f"Used metadata filter to delete vectors for {file_id}")
        return True
    except Exception as e:
        console.print(f"[red]Metadata filter deletion failed: {e}[/red]")
        return False
    
# -------------------------------
# 10) Poll & Update: 
#     Checks for new/changed/deleted files in your Drive folder.
# -------------------------------
def poll_drive_folder():
    """
    Return a list of files in the target folder. 
    Returns an empty list on error or if none found.
    """
    try:
        results = drive_service.files().list(
            q=f"'{GOOGLE_DRIVE_FOLDER_ID}' in parents and trashed = false",
            fields="files(id, name, modifiedTime)"
        ).execute()
        return results.get("files", [])
    except Exception as e:
        console.print(f"[red]Drive polling error: {e}[/red]")
        return []

def update_files():
    """
    Main logic to sync files from Google Drive into Pinecone.
    - Detect removed files (delete vectors).
    - For new or modified files, re-process and upsert.
    """
    console.print(f"\n=== Update started {datetime.now().isoformat()} ===\n")
    processed = load_processed_files()
    
    try:
        current_files = poll_drive_folder() or []
        current_ids = {f['id'] for f in current_files}

        # 1) Handle deletions
        for file_id in list(processed.keys()):
            if file_id not in current_ids:
                console.print(f"Removing vectors for file (no longer exists in Drive): {file_id}")
                if delete_vectors(file_id):
                    del processed[file_id]
                    save_processed_files(processed)
        
        # 2) Handle new or updated
        for file in current_files:
            existing = processed.get(file['id'])
            if (not existing) or (file['modifiedTime'] > existing['modified']):
                process_file(file)
    except Exception as e:
        console.print(f"Update failed: {str(e)}")
        raise

# -------------------------------
# Optional: A simple main loop
# -------------------------------
def wait_or_pull(interval=3600):
    """
    Wait for a specified interval, but allow typing 'pull' to do an immediate update
    or 'q' to quit.
    """
    start_time = time.time()
    while time.time() - start_time < interval:
        user_input = input("Type 'pull' to run update immediately or 'q' to quit: ").strip().lower()
        if user_input == "pull":
            return
        elif user_input == "q":
            print("Exiting...")
            sys.exit(0)
        time.sleep(1)

if __name__ == "__main__":
    while True:
        update_files()
        wait_or_pull()