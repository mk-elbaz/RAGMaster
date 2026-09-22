"""
Document parsers for multiple file formats.

This module demonstrates:
- Robust parsing with error handling
- Metadata extraction (filename, page numbers, sections)
- Clean text extraction with whitespace normalization
- Extensible parser architecture
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod
import re

from bs4 import BeautifulSoup
import PyPDF2
from docx import Document as DocxDocument


class ParsedDocument:
    """Container for parsed document with metadata."""
    
    def __init__(
        self,
        text: str,
        source_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.text = text
        self.source_id = source_id
        self.metadata = metadata or {}
        
    def __repr__(self) -> str:
        return f"ParsedDocument(source_id='{self.source_id}', length={len(self.text)})"


class DocumentParser(ABC):
    """Base class for document parsers."""
    
    @abstractmethod
    def can_parse(self, file_path: str) -> bool:
        """Check if this parser can handle the file."""
        pass
    
    @abstractmethod
    def parse(self, file_path: str) -> ParsedDocument:
        """Parse the document and return structured content."""
        pass
    
    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean and normalize text.
        
        - Remove excessive whitespace
        - Normalize line breaks
        - Remove special characters that don't add meaning
        """
        # Replace multiple spaces with single space
        text = re.sub(r' +', ' ', text)
        # Replace multiple newlines with double newline
        text = re.sub(r'\n\s*\n+', '\n\n', text)
        # Remove leading/trailing whitespace
        text = text.strip()
        return text


class PDFParser(DocumentParser):
    """
    Parse PDF documents.
    
    Extracts:
    - Full text content
    - Page numbers for each section
    - Metadata (title, author, page count)
    """
    
    def can_parse(self, file_path: str) -> bool:
        return file_path.lower().endswith('.pdf')
    
    def parse(self, file_path: str) -> ParsedDocument:
        """Parse PDF and extract text with page metadata."""
        text_parts = []
        metadata = {
            "file_type": "pdf",
            "file_path": file_path,
            "file_name": os.path.basename(file_path),
            "pages": []
        }
        
        try:
            # Try pymupdf first (much better text extraction)
            try:
                import pymupdf
                
                doc = pymupdf.open(file_path)
                metadata["page_count"] = len(doc)
                
                # Extract metadata
                pdf_metadata = doc.metadata
                if pdf_metadata:
                    metadata["title"] = pdf_metadata.get("title", "")
                    metadata["author"] = pdf_metadata.get("author", "")
                
                # Extract text from each page
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    page_text = page.get_text("text")  # Extract as plain text with proper spacing
                    
                    if page_text:
                        cleaned_text = self.clean_text(page_text)
                        if cleaned_text:
                            text_parts.append(f"[Page {page_num + 1}]\n{cleaned_text}")
                            metadata["pages"].append({
                                "page_num": page_num + 1,
                                "char_count": len(cleaned_text)
                            })
                
                doc.close()
                
            except ImportError:
                # Fallback to PyPDF2
                print("⚠️  pymupdf not available, using PyPDF2 (may have spacing issues)")
                with open(file_path, "rb") as f:
                    reader = PyPDF2.PdfReader(f)
                    metadata["page_count"] = len(reader.pages)
                    
                    # Extract PDF metadata
                    if reader.metadata:
                        metadata["title"] = reader.metadata.get("/Title", "")
                        metadata["author"] = reader.metadata.get("/Author", "")
                    
                    # Extract text from each page
                    for page_num, page in enumerate(reader.pages, start=1):
                        page_text = page.extract_text()
                        if page_text:
                            cleaned_text = self.clean_text(page_text)
                            if cleaned_text:  # Only add non-empty pages
                                text_parts.append(f"[Page {page_num}]\n{cleaned_text}")
                                metadata["pages"].append({
                                    "page_num": page_num,
                                    "char_count": len(cleaned_text)
                                })
            
            full_text = "\n\n".join(text_parts)
            source_id = f"pdf:{os.path.basename(file_path)}"
            
            return ParsedDocument(full_text, source_id, metadata)
            
        except Exception as e:
            raise ValueError(f"Failed to parse PDF {file_path}: {str(e)}")


class HTMLParser(DocumentParser):
    """
    Parse HTML documents.
    
    Extracts:
    - Main content (removes scripts, styles, navigation)
    - Heading hierarchy
    - Link references
    """
    
    def can_parse(self, file_path: str) -> bool:
        ext = file_path.lower()
        return ext.endswith('.html') or ext.endswith('.htm')
    
    def parse(self, file_path: str) -> ParsedDocument:
        """Parse HTML and extract clean text."""
        metadata = {
            "file_type": "html",
            "file_path": file_path,
            "file_name": os.path.basename(file_path),
            "headings": []
        }
        
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                soup = BeautifulSoup(f, "html.parser")
                
                # Extract title
                if soup.title:
                    metadata["title"] = soup.title.string
                
                # Remove unwanted elements
                for element in soup(["script", "style", "nav", "footer", "header"]):
                    element.decompose()
                
                # Extract headings for structure
                for heading in soup.find_all(['h1', 'h2', 'h3']):
                    metadata["headings"].append({
                        "level": heading.name,
                        "text": heading.get_text(strip=True)
                    })
                
                # Get main text
                text = soup.get_text(separator="\n")
                cleaned_text = self.clean_text(text)
                
                source_id = f"html:{os.path.basename(file_path)}"
                
                return ParsedDocument(cleaned_text, source_id, metadata)
                
        except Exception as e:
            raise ValueError(f"Failed to parse HTML {file_path}: {str(e)}")


class DOCXParser(DocumentParser):
    """
    Parse Microsoft Word documents (.docx).
    
    Extracts:
    - Paragraphs with formatting
    - Heading structure
    - Tables (as text)
    """
    
    def can_parse(self, file_path: str) -> bool:
        return file_path.lower().endswith('.docx')
    
    def parse(self, file_path: str) -> ParsedDocument:
        """Parse DOCX and extract text."""
        metadata = {
            "file_type": "docx",
            "file_path": file_path,
            "file_name": os.path.basename(file_path),
            "headings": []
        }
        
        try:
            doc = DocxDocument(file_path)
            text_parts = []
            
            # Extract paragraphs
            for para in doc.paragraphs:
                if para.text.strip():
                    # Check if it's a heading
                    if para.style.name.startswith('Heading'):
                        metadata["headings"].append({
                            "level": para.style.name,
                            "text": para.text.strip()
                        })
                    text_parts.append(para.text.strip())
            
            # Extract tables
            for table in doc.tables:
                table_text = []
                for row in table.rows:
                    row_text = " | ".join(cell.text.strip() for cell in row.cells)
                    if row_text.strip():
                        table_text.append(row_text)
                if table_text:
                    text_parts.append("\n".join(table_text))
            
            full_text = self.clean_text("\n\n".join(text_parts))
            source_id = f"docx:{os.path.basename(file_path)}"
            
            metadata["paragraph_count"] = len(doc.paragraphs)
            metadata["table_count"] = len(doc.tables)
            
            return ParsedDocument(full_text, source_id, metadata)
            
        except Exception as e:
            raise ValueError(f"Failed to parse DOCX {file_path}: {str(e)}")


class TextParser(DocumentParser):
    """
    Parse plain text files.
    
    Simple parser for .txt, .md files.
    """
    
    def can_parse(self, file_path: str) -> bool:
        ext = file_path.lower()
        return ext.endswith('.txt') or ext.endswith('.md')
    
    def parse(self, file_path: str) -> ParsedDocument:
        """Parse plain text file."""
        metadata = {
            "file_type": "text",
            "file_path": file_path,
            "file_name": os.path.basename(file_path)
        }
        
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            
            cleaned_text = self.clean_text(text)
            source_id = f"text:{os.path.basename(file_path)}"
            
            # Basic statistics
            metadata["line_count"] = len(text.splitlines())
            metadata["char_count"] = len(text)
            
            return ParsedDocument(cleaned_text, source_id, metadata)
            
        except Exception as e:
            raise ValueError(f"Failed to parse text file {file_path}: {str(e)}")


class ConfluenceParser(DocumentParser):
    """
    Parse Confluence pages exported as JSON.
    
    For MVP, expects JSON format:
    {
        "id": "page_id",
        "title": "Page Title",
        "body_text": "Page content...",
        "url": "https://..."
    }
    
    Later: Integrate with Confluence REST API for live fetching.
    """
    
    def can_parse(self, file_path: str) -> bool:
        return file_path.lower().endswith('.json')
    
    def parse(self, file_path: str) -> ParsedDocument:
        """Parse Confluence JSON export."""
        metadata = {
            "file_type": "confluence",
            "file_path": file_path,
            "file_name": os.path.basename(file_path)
        }
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Extract content
            text = data.get("body_text", "")
            metadata["title"] = data.get("title", "")
            metadata["confluence_id"] = data.get("id", "")
            metadata["url"] = data.get("url", "")
            
            cleaned_text = self.clean_text(text)
            source_id = f"confluence:{data.get('id', os.path.basename(file_path))}"
            
            return ParsedDocument(cleaned_text, source_id, metadata)
            
        except Exception as e:
            raise ValueError(f"Failed to parse Confluence JSON {file_path}: {str(e)}")


class DocumentParserFactory:
    """
    Factory for getting the right parser for a file.
    
    This demonstrates the Factory pattern - select parser based on file type.
    """
    
    def __init__(self):
        self.parsers: List[DocumentParser] = [
            PDFParser(),
            HTMLParser(),
            DOCXParser(),
            TextParser(),
            ConfluenceParser()
        ]
    
    def get_parser(self, file_path: str) -> Optional[DocumentParser]:
        """Get the appropriate parser for a file."""
        for parser in self.parsers:
            if parser.can_parse(file_path):
                return parser
        return None
    
    def parse_file(self, file_path: str) -> Optional[ParsedDocument]:
        """Parse a file using the appropriate parser."""
        parser = self.get_parser(file_path)
        if parser:
            try:
                return parser.parse(file_path)
            except Exception as e:
                print(f"Error parsing {file_path}: {e}")
                return None
        else:
            print(f"No parser available for {file_path}")
            return None
    
    def parse_directory(self, directory: str, recursive: bool = True) -> List[ParsedDocument]:
        """
        Parse all supported files in a directory.
        
        Args:
            directory: Path to directory
            recursive: Whether to scan subdirectories
        
        Returns:
            List of successfully parsed documents
        """
        documents = []
        path = Path(directory)
        
        # Get all files
        if recursive:
            files = path.rglob("*")
        else:
            files = path.glob("*")
        
        # Filter to files only
        files = [f for f in files if f.is_file()]
        
        print(f"Found {len(files)} files in {directory}")
        
        for file_path in files:
            doc = self.parse_file(str(file_path))
            if doc:
                documents.append(doc)
                print(f"✓ Parsed {file_path.name}")
        
        print(f"\nSuccessfully parsed {len(documents)} documents")
        return documents
