# Lab 14: Document Loading using LangChain for RAG

## Lab Overview
Introduction to Document Loaders in LangChain for Retrieval-Augmented Generation systems.

## Objectives
- Understand RAG architecture
- Use various LangChain document loaders
- Inspect document metadata
- Prepare documents for retrieval

## Tasks Completed

### Task 1: Environment Setup
- Created virtual environment
- Installed LangChain libraries
- Verified installation

### Task 2: Understanding Document Loaders
- Studied role in RAG pipeline
- Learned Document object structure
- Understood page_content and metadata

### Task 3: PDF Loading (PyPDFLoader)
- Loaded dl-curriculum.pdf
- Counted total pages
- Displayed first page content
- Inspected metadata

### Task 4: CSV Loading (CSVLoader)
- Loaded Social_Network_Ads.csv
- Inspected row-to-document conversion
- Printed sample documents

### Task 5: Text Loading (TextLoader)
- Loaded cricket.txt
- Examined plain text handling
- Compared with PDF loader

### Task 6: Comparison Table
- Compared all loaders
- Analyzed content formats
- Documented metadata fields

## Libraries Used
```python
from langchain_community.document_loaders import PyPDFLoader, CSVLoader, TextLoader, WebBaseLoader
import pandas as pd
```

## Key Learnings
- Document loaders convert raw data to Document objects
- Each loader handles specific file formats
- Metadata crucial for source tracking
- RAG pipeline preparation
- Importance of document quality
- Directory-based loading for scalability