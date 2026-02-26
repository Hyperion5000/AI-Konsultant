import re
from typing import List, Optional, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

class LegalDocumentParser:
    """
    A custom parser for legal documents that preserves context (Chapter, Article)
    across chunks by treating them as a state machine.
    """
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # We use a standard splitter for the content within a specific article context
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
            is_separator_regex=False
        )

        # Regex patterns
        # Matches: "Глава I", "Глава 1", "Глава 5" (at start of line, allowing whitespace)
        # Capture group 1 is the full "Глава X" string to be used as metadata
        self.chapter_pattern = re.compile(r"^\s*(Глава\s+(?:[0-9IVX]+|\d+))", re.IGNORECASE)

        # Matches: "Статья 1", "Ст. 1", "Статья 1.1" (at start of line, allowing whitespace)
        # Capture group 1 is the full "Статья X" string
        self.article_pattern = re.compile(r"^\s*((?:Статья|Ст\.?)\s+\d+(?:\.\d+)*)", re.IGNORECASE)

    def parse(self, text: str, filename: str) -> List[Document]:
        """
        Parses the text, tracking current chapter and article, and yields Documents
        with metadata.
        """
        documents = []
        lines = text.splitlines()

        current_chapter: Optional[str] = None
        current_article: Optional[str] = None

        buffer: List[str] = []

        for line in lines:
            chapter_match = self.chapter_pattern.match(line)
            article_match = self.article_pattern.match(line)

            is_new_context = False

            if chapter_match:
                # Flush previous context
                self._flush_buffer(buffer, filename, current_chapter, current_article, documents)
                buffer = []

                # Update state
                current_chapter = chapter_match.group(1).strip()
                current_article = None # Reset article on new chapter
                is_new_context = True

            elif article_match:
                # Flush previous context
                self._flush_buffer(buffer, filename, current_chapter, current_article, documents)
                buffer = []

                # Update state
                current_article = article_match.group(1).strip()
                is_new_context = True

            # Add line to buffer (including the header line itself)
            buffer.append(line)

        # Flush remaining buffer
        if buffer:
            self._flush_buffer(buffer, filename, current_chapter, current_article, documents)

        return documents

    def _flush_buffer(
        self,
        buffer: List[str],
        filename: str,
        chapter: Optional[str],
        article: Optional[str],
        documents: List[Document]
    ):
        if not buffer:
            return

        text_content = "\n".join(buffer).strip()
        if not text_content:
            return

        # Prepare metadata
        metadata: Dict[str, Any] = {"source": filename}
        if chapter:
            metadata["chapter"] = chapter
        if article:
            metadata["article"] = article

        # Split the buffered text into chunks
        # Since this buffer belongs to a single semantic context (same Article/Chapter),
        # we can safely split it using standard logic without losing metadata.
        split_texts = self.text_splitter.split_text(text_content)

        for i, chunk_text in enumerate(split_texts):
            doc_metadata = metadata.copy()
            # Optional: Add chunk index within the article?
            # doc_metadata["chunk_index"] = i
            documents.append(Document(page_content=chunk_text, metadata=doc_metadata))
