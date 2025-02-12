# Google Drive to Pinecone Vector Indexer

This project integrates Google Drive with Pinecone, enabling real-time indexing of text files from a specified folder. It uses Google's Generative AI for text summarization and Pinecone for vector storage and search.

## Key Features

- Automatically indexes text files from a Google Drive folder.
- Uses Google's Generative AI for text summarization.
- Stores file metadata and summarized text vectors in Pinecone.
- Supports incremental updates by tracking processed files.

## Installation

1. Install Python 3.8 or higher.
2. Create a new virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Create a `.env` file in the project root directory with the following content:
   ```
   GOOGLE_APPLICATION_CREDENTIALS=path/to/your/service-account-file.json
   PINECONE_API_KEY=your-pinecone-api-key
   PINECONE_INDEX_NAME=your-pinecone-index-name
   DRIVE_FOLDER_ID=your-google-drive-folder-id
   GROQ_API_KEY=your-groq-api-key
   ```

## Usage

1. Run the main indexing script:
   ```
   python main.py
   ```
2. Run the vector updater script to update existing files:
   ```
   python vector_updater.py
   ```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License.

## Acknowledgments

- [Google Drive API](https://developers.google.com/drive/api)
- [Pinecone](https://www.pinecone.io/)
- [Google Generative AI](https://generativeai.google/)
- [Groq Python client](https://github.com/groq-lang/groq-python)
- [Rich](https://github.com/willmcgugan/rich) for console output
- [python-dotenv](https://github.com/theskumar/python-dotenv) for managing environment variables

## Badges

[![MIT License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Note: This README is a brief overview, and the actual project may have additional features or documentation.