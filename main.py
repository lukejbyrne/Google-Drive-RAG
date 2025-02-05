import os
import time
import json
import io
import requests
import numpy as np
from groq import Groq  # Using official Groq Python client
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.console import Console
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import google.generativeai as palm  # Official Google Generative AI client

# -------------------------------
# Configuration & Initialization
# -------------------------------

console = Console()

# Environment Variables (set these in your environment or update here)
SERVICE_ACCOUNT_FILE = os.environ.get("GOOGLE_DRIVE_SERVICE_ACCOUNT_FILE")  # e.g. "path/to/your/service-account.json"
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_ENV = os.environ.get("PINECONE_ENV")  # e.g. "us-west-2"
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "company-files")
GOOGLE_GEMINI_API_KEY = os.environ.get("GOOGLE_GEMINI_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GOOGLE_DRIVE_FOLDER_ID = os.environ.get("GOOGLE_DRIVE_FOLDER_ID")

if not all([SERVICE_ACCOUNT_FILE, PINECONE_API_KEY, PINECONE_ENV, GOOGLE_GEMINI_API_KEY, GROQ_API_KEY, GOOGLE_DRIVE_FOLDER_ID]):
    console.print("[red]Error: Missing one or more required environment variables.[/red]")
    exit(1)

# Initialize Google Drive API using a service account
SCOPES = ['https://www.googleapis.com/auth/drive']
credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
drive_service = build('drive', 'v3', credentials=credentials)

# -------------------------------
# Initialize Pinecone using the new client instance
# -------------------------------
from pinecone import Pinecone, ServerlessSpec

# Create a Pinecone client instance
pc = Pinecone(api_key=PINECONE_API_KEY)
# List indexes to check if the index exists; if not, create it.
existing_indexes = [idx["name"] for idx in pc.list_indexes()]
if PINECONE_INDEX_NAME not in existing_indexes:
    console.print(f"[yellow]Index '{PINECONE_INDEX_NAME}' not found. Creating new index...[/yellow]")
    # Adjust "dimension" (e.g. 768) to match the output dimension of your Gemini embedding.
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=PINECONE_ENV)
    )
# Obtain the index instance (using the namespace "default")
index = pc.Index(PINECONE_INDEX_NAME)

# Configure Google Gemini (PaLM)
palm.configure(api_key=GOOGLE_GEMINI_API_KEY)

# -------------------------------
# Processed Files Tracking Updates
# -------------------------------

# File used to track processed files (to avoid re‑processing)
PROCESSED_FILES_PATH = "processed_files.json"

# Update the processed_files functions to use a dictionary structure:
def load_processed_files():
    """
    Returns a dictionary mapping file_id to its last processed modifiedTime.
    """
    if os.path.exists(PROCESSED_FILES_PATH):
        with open(PROCESSED_FILES_PATH, "r") as f:
            return json.load(f)
    else:
        return {}  # file_id -> modifiedTime

def save_processed_files(processed):
    with open(PROCESSED_FILES_PATH, "w") as f:
        json.dump(processed, f, indent=2)


# -------------------------------
# Google Drive Download Function
# -------------------------------

def download_file(file_id: str, file_name: str) -> str:
    """
    Downloads a file from Google Drive using its file ID.
    Returns the file content as a UTF-8 string.
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
            progress.update(task, completed=int(status.progress() * 100))
    fh.seek(0)
    try:
        content = fh.read().decode("utf-8")
    except Exception as e:
        console.print(f"[red]Error decoding file content: {e}[/red]")
        content = ""
    return content


# -------------------------------
# Text Splitting Function
# -------------------------------

def split_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list:
    """
    Splits text into chunks with a specified chunk size and overlap.
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
# Embedding Function via Google Gemini (using embed_content)
# -------------------------------

def get_embedding(text: str) -> list:
    """
    Calls the Google Gemini (PaLM) API to compute an embedding.
    Uses the embed_content function (instead of embed_text).
    Assumes the response contains an "embedding" key.
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
# Process File: Download, Split, Embed & Upsert to Pinecone
# -------------------------------

def process_file(file: dict):
    console.rule(f"[bold blue]Processing File: {file['name']}")
    content = download_file(file["id"], file["name"])
    if not content:
        console.print(f"[red]No content downloaded for file: {file['name']}[/red]")
        return
    console.print("[green]File downloaded. Splitting text...[/green]")
    chunks = split_text(content)
    console.print(f"[green]Text split into {len(chunks)} chunks. Processing chunks...[/green]")
    
    with Progress(
        "[progress.description]{task.description}",
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        transient=True,
    ) as progress:
        task = progress.add_task("Embedding & Upserting chunks", total=len(chunks))
        for i, chunk in enumerate(chunks):
            embedding = get_embedding(chunk)
            if embedding is None:
                continue
            # Create a unique vector ID based on the file and chunk index
            vector_id = f"{file['id']}_{i}"
            metadata = {
                "file_id": file["id"],
                "file_name": file["name"],
                "chunk_index": i,
                "text": chunk[:200]  # store a preview (first 200 characters)
            }
            try:
                index.upsert(
                    vectors=[{"id": vector_id, "values": embedding, "metadata": metadata}],
                    namespace="default"
                )
            except Exception as e:
                console.print(f"[red]Error upserting vector: {e}[/red]")
            time.sleep(0.1)  # slight delay to avoid rate limits
            progress.advance(task)
    console.print("[bold green]File processing complete and added to Pinecone vector store.[/bold green]\n")


# -------------------------------
# Query Pinecone for Similar Vectors
# -------------------------------

def search_pinecone(query_embedding: list, top_k: int = 3):
    try:
        result = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            namespace="default"
        )
        return result
    except Exception as e:
        console.print(f"[red]Error querying Pinecone: {e}[/red]")
        return None


# -------------------------------
# Call Groq Chat Model API for Chat Completion
# -------------------------------

def groq_chat(system_message: str, query: str, context: str) -> str:
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": f"{system_message}\n\nContext: {context}"
                },
                {
                    "role": "user",
                    "content": query
                }
            ],
            model="deepseek-r1-distill-llama-70b",
            temperature=0.2,
            max_tokens=512
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"
      

# -------------------------------
# AI Agent: Retrieve Context & Answer Query
# -------------------------------

def chat_agent(query: str) -> str:
    system_message = (
        "You are a helpful HR assistant designed to answer employee questions based on company policies. "
        "Retrieve relevant information from the provided internal documents and provide a concise, accurate, and informative answer. "
        "Use the tool called 'company_documents_tool' to retrieve any information from the company's documents. "
        "If the answer cannot be found in the provided documents, respond with 'I cannot find the answer in the available resources.'"
    )
    query_embedding = get_embedding(query)
    if query_embedding is None:
        return "Error obtaining query embedding."
    results = search_pinecone(query_embedding, top_k=3)
    if results is None or "matches" not in results or not results["matches"]:
        return "I cannot find the answer in the available resources."
    # Combine retrieved texts from the matches for context
    matches = results["matches"]
    context = " ".join(match["metadata"].get("text", "") for match in matches)
    if not context.strip():
        return "I cannot find the answer in the available resources."
    answer = groq_chat(system_message, query, context)
    return answer


# -------------------------------
# Poll Google Drive Folder for Files
# -------------------------------

def poll_drive_folder():
    """
    Polls the Google Drive folder and ensures that the Pinecone vector store 
    reflects the exact state of the folder. It handles:
      - New files: processes and adds vectors.
      - Updated files: deletes old vectors and reprocesses.
      - Deleted files: deletes vectors associated with files that no longer exist.
    """
    # Build the query to list all non-trashed files in the target folder
    query = f"'{GOOGLE_DRIVE_FOLDER_ID}' in parents and trashed = false"
    try:
        results = drive_service.files().list(
            q=query, fields="files(id, name, modifiedTime)"
        ).execute()
        files = results.get("files", [])
        if not files:
            console.print("[yellow]No files found in the folder.[/yellow]")
    except Exception as e:
        console.print(f"[red]Error polling Google Drive folder: {e}[/red]")
        return

    # Load the dictionary of processed files (file_id -> modifiedTime)
    processed_files = load_processed_files()
    # Build a set of file IDs currently in the folder
    current_file_ids = {file["id"] for file in files}

    # Process new or updated files
    for file in files:
        file_id = file["id"]
        current_modified_time = file["modifiedTime"]

        # Determine if this is a new or updated file.
        if file_id not in processed_files or current_modified_time > processed_files[file_id]:
            if file_id in processed_files:
                console.print(f"[blue]Updated file detected: {file['name']} (ID: {file_id}). Reprocessing...[/blue]")
                # Delete previous vectors for this file from Pinecone.
                try:
                    index.delete(filter={"file_id": file_id}, namespace="default")
                    console.print(f"[yellow]Deleted old vectors for file ID: {file_id}[/yellow]")
                except Exception as e:
                    console.print(f"[red]Error deleting old vectors for file ID {file_id}: {e}[/red]")
            else:
                console.print(f"[green]New file detected: {file['name']} (ID: {file_id})[/green]")
            
            # Process the file (download, split, embed, and upsert vectors)
            process_file(file)
            # Record the file's current modifiedTime
            processed_files[file_id] = current_modified_time

    # Handle deleted files: find file IDs that were processed previously but are now missing
    deleted_file_ids = set(processed_files.keys()) - current_file_ids
    for file_id in deleted_file_ids:
        console.print(f"[magenta]File deleted (ID: {file_id}). Removing its vectors.[/magenta]")
        try:
            index.delete(filter={"file_id": file_id}, namespace="default")
        except Exception as e:
            console.print(f"[red]Error deleting vectors for deleted file ID {file_id}: {e}[/red]")
        # Remove the file entry from the processed files dictionary
        del processed_files[file_id]

    # Save the updated processed_files dictionary
    save_processed_files(processed_files)