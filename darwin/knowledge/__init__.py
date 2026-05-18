"""Knowledge Base package.

Provides ingestion, indexing, and retrieval of user-supplied reference
materials (text files, Markdown, plain PDFs, subtitle tracks for video,
and raw video metadata).

Everything is stored **locally** in ``data/knowledge/`` – no cloud
service, no API key, no paid subscription required.
"""

from darwin.knowledge.base import KnowledgeBase, KnowledgeEntry
from darwin.knowledge.ingestor import Ingestor
from darwin.knowledge.retriever import Retriever

__all__ = ["KnowledgeBase", "KnowledgeEntry", "Ingestor", "Retriever"]
