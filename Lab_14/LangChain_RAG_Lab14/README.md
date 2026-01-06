## Lab 14 - Document Loading with LangChain

## Lab Objective
To understand Document Loaders in LangChain and prepare documents for Retrieval-Augmented Generation (RAG) systems.

## Topics Covered

- Document Loaders in LangChain
- Loading data from multiple formats
- Document metadata inspection
- Text preprocessing
- RAG system fundamentals
- Comparing different loaders

## What is RAG?
Retrieval-Augmented Generation combines:

Retrieval: Finding relevant documents
Generation: Using LLM to generate answers

Document loaders are the first step in this process.

## Lab Tasks Completed

- Task 1: Environment Setup
bashpip install langchain langchain-community pypdf unstructured

- Task 2: Understanding Document Loaders
Concept:
Document loaders convert raw data into LangChain Document objects.
Document Structure:
pythonDocument(
    page_content="extracted text...",
    metadata={"source": "file.pdf", "page": 1}
)

- Task 3: Load PDF Data (PyPDFLoader)
pythonfrom langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('dl-curriculum.pdf')
docs = loader.load()

print(len(docs))              # Total pages
print(docs[0].page_content)   # First page text
print(docs[1].metadata)       # Metadata
Results:

Loaded PDF page by page
Each page is separate document
Metadata includes page number and source

- Task 4: Structured Data (CSVLoader)
pythonfrom langchain_community.document_loaders import CSVLoader

loader = CSVLoader(file_path='Social_Network_Ads.csv')
docs = loader.load()

print(len(docs))      # Number of rows
print(docs[1])        # Sample document
Results:

Each CSV row becomes a document
Metadata includes file path and row index
Text format: "column: value\ncolumn: value"

- Task 5: Load Web Content (WebBaseLoader)
pythonfrom langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://www.langchain.com")
docs = loader.load()

print(docs[0].page_content[:200])
print(docs[0].metadata)
Results:

Extracted clean text from webpage
Metadata includes URL
Removed HTML tags automatically

- Task 6: Compare All Loaders
pythonloaders = {
    "PDF": PyPDFLoader("dl-curriculum.pdf"),
    "CSV": CSVLoader("Social_Network_Ads.csv"),
    "WEB": WebBaseLoader("https://www.langchain.com"),
    "TEXT": TextLoader("cricket.txt")
}

for name, loader in loaders.items():
    docs = loader.load()
    print(f"\n===== {name} Document Loader =====")
    print("Sample text:", docs[0].page_content[:200])

    print("Metadata:", docs[0].metadata)



## Metadata helps:

Track source
Identify page numbers
Improve accuracy
Enable traceability

3. TextLoader vs PyPDFLoader
Answer:

TextLoader: Plain text files
PyPDFLoader: PDF files, page by page

4. Scanned PDF Images
Answer:
Text cannot be extracted without OCR. Content will be empty or unreadable.
5. Directory-Based Loading
Answer:
Allows loading multiple files automatically. Useful for large document collections.
6. Document Quality Effect
Answer:
Clean, accurate documents → better retrieval → more correct answers.
Libraries Used
pythonfrom langchain_community.document_loaders import (
    PyPDFLoader,
    CSVLoader,
    WebBaseLoader,
    TextLoader
)

## Key Learnings

Document Object Structure
pythonclass Document:
    page_content: str      # Extracted text
    metadata: dict         # Source information
Metadata Fields by Loader

PyPDFLoader: source, page
WebBaseLoader: source (URL)
CSVLoader: source, row
TextLoader: source

## Best Practices

Choose appropriate loader for file type
Inspect metadata to verify extraction
Handle encoding errors (UTF-8, CP1252)
Check content quality before using
Use DirectoryLoader for multiple files

RAG Pipeline
1. Load Documents (This Lab)
   ↓
2. Split into Chunks
   ↓
3. Create Embeddings
   ↓
4. Store in Vector Database
   ↓
5. Retrieve Relevant Chunks
   ↓
6. Generate Answer with LLM


## Files:

- Lab_14.pdf
- dl-curriculum.pdf
- Social_Network_Ads.csv
- cricket.txt

