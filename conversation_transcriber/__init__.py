"""
Conversation Transcriber

AI-powered conversation transcription with speaker separation and intelligent summarization.
"""

__version__ = "0.1.0"
__author__ = "shaopei"

from .conversation_transcriber import main as transcribe
from .batch_transcribe import main as batch_transcribe

__all__ = ["transcribe", "batch_transcribe"] 