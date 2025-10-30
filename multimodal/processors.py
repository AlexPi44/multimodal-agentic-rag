"""Advanced multimodal document processors for various file types."""

import asyncio
import os
import mimetypes
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import hashlib

# Core processing libraries
import PyPDF2
import docx
from PIL import Image
import cv2
import numpy as np
import whisper
import librosa
import moviepy.editor as mp

# Code and markup processing
import ast
import html
from bs4 import BeautifulSoup
import markdown
import json
import xml.etree.ElementTree as ET

# Text extraction
import textract
import chardet

from src.core.config import settings
from src.core.logging import get_logger

logger = get_logger(__name__)


class DocumentProcessor:
    """Advanced document processor for multimodal content."""
    
    def __init__(self):
        """Initialize the document processor."""
        self.supported_extensions = {
            # Text documents
            '.txt': 'text',
            '.md': 'markdown',
            '.markdown': 'markdown',
            '.rst': 'text',
            
            # Office documents
            '.pdf': 'pdf',
            '.docx': 'docx',
            '.doc': 'doc',
            '.pptx': 'presentation',
            '.xlsx': 'spreadsheet',
            
            # Code files
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.jsx': 'react',
            '.tsx': 'react',
            '.html': 'html',
            '.htm': 'html',
            '.css': 'css',
            '.scss': 'scss',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.h': 'c_header',
            '.go': 'go',
            '.rs': 'rust',
            '.php': 'php',
            '.rb': 'ruby',
            '.swift': 'swift',
            '.kt': 'kotlin',
            '.sql': 'sql',
            '.sh': 'shell',
            '.bash': 'shell',
            '.yml': 'yaml',
            '.yaml': 'yaml',
            '.toml': 'toml',
            '.json': 'json',
            '.xml': 'xml',
            
            # Images
            '.jpg': 'image',
            '.jpeg': 'image',
            '.png': 'image',
            '.gif': 'image',
            '.bmp': 'image',
            '.tiff': 'image',
            '.webp': 'image',
            '.svg': 'svg',
            
            # Audio
            '.mp3': 'audio',
            '.wav': 'audio',
            '.flac': 'audio',
            '.aac': 'audio',
            '.ogg': 'audio',
            '.m4a': 'audio',
            
            # Video
            '.mp4': 'video',
            '.avi': 'video',
            '.mov': 'video',
            '.wmv': 'video',
            '.flv': 'video',
            '.webm': 'video',
            
            # Archives (for extraction)
            '.zip': 'archive',
            '.tar': 'archive',
            '.gz': 'archive',
            '.rar': 'archive',
        }
        
        logger.info(f"Initialized document processor with {len(self.supported_extensions)} supported file types")
    
    async def process_file(
        self, 
        file_path: Union[str, Path], 
        document_type: str = "auto",
        chunk_size: int = 1000,
        overlap: int = 200
    ) -> List[Dict[str, Any]]:
        """Process a file and extract content.
        
        Args:
            file_path: Path to the file to process
            document_type: Type of document (auto-detected if "auto")
            chunk_size: Size of text chunks
            overlap: Overlap between chunks
            
        Returns:
            List of processed document chunks
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Auto-detect document type
        if document_type == "auto":
            document_type = self._detect_file_type(file_path)
        
        logger.info(f"Processing file: {file_path} (type: {document_type})")
        
        try:
            # Extract content based on file type
            if document_type in ['text', 'markdown']:
                content = await self._process_text_file(file_path)
                modality = 'text'
            elif document_type == 'pdf':
                content = await self._process_pdf(file_path)
                modality = 'text'
            elif document_type in ['docx', 'doc']:
                content = await self._process_office_document(file_path)
                modality = 'text'
            elif document_type == 'html':
                content = await self._process_html(file_path)
                modality = 'text'
            elif document_type in ['python', 'javascript', 'typescript', 'java', 'cpp', 'c', 'go', 'rust', 'php', 'ruby', 'swift', 'kotlin']:
                content = await self._process_code_file(file_path, document_type)
                modality = 'text'
            elif document_type in ['json', 'xml', 'yaml', 'toml']:
                content = await self._process_structured_data(file_path, document_type)
                modality = 'text'
            elif document_type == 'image':
                content = await self._process_image(file_path)
                modality = 'image'
            elif document_type == 'audio':
                content = await self._process_audio(file_path)
                modality = 'audio'
            elif document_type == 'video':
                content = await self._process_video(file_path)
                modality = 'multimodal'  # Video contains both audio and visual
            else:
                # Fallback to text extraction
                content = await self._process_generic_file(file_path)
                modality = 'text'
            
            # Create document metadata
            file_stats = file_path.stat()
            metadata = {
                'filename': file_path.name,
                'file_path': str(file_path),
                'file_size': file_stats.st_size,
                'file_type': document_type,
                'modality': modality,
                'created_at': datetime.fromtimestamp(file_stats.st_ctime).isoformat(),
                'modified_at': datetime.fromtimestamp(file_stats.st_mtime).isoformat(),
                'processed_at': datetime.now().isoformat(),
                'file_hash': await self._calculate_file_hash(file_path)
            }
            
            # Split content into chunks if it's text
            if modality == 'text' and isinstance(content, str):
                chunks = self._split_text_into_chunks(content, chunk_size, overlap)
                
                documents = []
                for i, chunk in enumerate(chunks):
                    doc = {
                        'content': chunk,
                        'chunk_index': i,
                        'total_chunks': len(chunks),
                        'modality': modality,
                        'metadata': metadata.copy()
                    }
                    doc['metadata']['chunk_id'] = f"{metadata['file_hash']}_{i}"
                    documents.append(doc)
                
                return documents
            else:
                # For non-text content, return as single document
                return [{
                    'content': content,
                    'chunk_index': 0,
                    'total_chunks': 1,
                    'modality': modality,
                    'metadata': metadata
                }]
                
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            raise
    
    def _detect_file_type(self, file_path: Path) -> str:
        """Detect file type based on extension."""
        extension = file_path.suffix.lower()
        return self.supported_extensions.get(extension, 'text')
    
    async def _process_text_file(self, file_path: Path) -> str:
        """Process plain text and markdown files."""
        try:
            # Detect encoding
            with open(file_path, 'rb') as f:
                raw_data = f.read()
                encoding = chardet.detect(raw_data)['encoding'] or 'utf-8'
            
            # Read with detected encoding
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()
            
            # If markdown, convert to plain text while preserving structure
            if file_path.suffix.lower() in ['.md', '.markdown']:
                # Convert markdown to HTML then extract text
                html_content = markdown.markdown(content)
                soup = BeautifulSoup(html_content, 'html.parser')
                content = soup.get_text(separator='\n', strip=True)
            
            return content
            
        except Exception as e:
            logger.error(f"Error processing text file: {e}")
            return f"Error reading file: {str(e)}"
    
    async def _process_pdf(self, file_path: Path) -> str:
        """Process PDF files."""
        try:
            content = ""
            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    page_text = page.extract_text()
                    content += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
            
            return content.strip()
            
        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            return f"Error reading PDF: {str(e)}"
    
    async def _process_office_document(self, file_path: Path) -> str:
        """Process Office documents (Word, PowerPoint, Excel)."""
        try:
            if file_path.suffix.lower() == '.docx':
                doc = docx.Document(file_path)
                content = "\n".join([paragraph.text for paragraph in doc.paragraphs])
                return content
            else:
                # Use textract for other formats
                content = await asyncio.to_thread(textract.process, str(file_path))
                return content.decode('utf-8')
                
        except Exception as e:
            logger.error(f"Error processing office document: {e}")
            return f"Error reading document: {str(e)}"
    
    async def _process_html(self, file_path: Path) -> str:
        """Process HTML files."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            # Parse HTML and extract text while preserving code structure
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Extract code blocks separately
            code_blocks = []
            for code_tag in soup.find_all(['code', 'pre', 'script', 'style']):
                if code_tag.name in ['script', 'style']:
                    code_blocks.append(f"\n--- {code_tag.name.upper()} BLOCK ---\n{code_tag.get_text()}\n")
                else:
                    code_blocks.append(f"\n--- CODE ---\n{code_tag.get_text()}\n")
                code_tag.decompose()  # Remove from main content
            
            # Get main text content
            main_text = soup.get_text(separator='\n', strip=True)
            
            # Combine main text with code blocks
            full_content = main_text + "\n".join(code_blocks)
            
            return full_content
            
        except Exception as e:
            logger.error(f"Error processing HTML: {e}")
            return f"Error reading HTML: {str(e)}"
    
    async def _process_code_file(self, file_path: Path, language: str) -> str:
        """Process code files with syntax analysis."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code_content = f.read()
            
            # Add language-specific analysis
            analysis = f"--- {language.upper()} FILE: {file_path.name} ---\n\n"
            
            if language == 'python':
                # Parse Python AST for additional structure info
                try:
                    tree = ast.parse(code_content)
                    functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
                    classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
                    
                    if functions:
                        analysis += f"Functions: {', '.join(functions)}\n"
                    if classes:
                        analysis += f"Classes: {', '.join(classes)}\n"
                    analysis += "\n"
                except:
                    pass  # Skip AST analysis if parsing fails
            
            # Count lines and add basic metrics
            lines = code_content.split('\n')
            non_empty_lines = [line for line in lines if line.strip()]
            comment_lines = [line for line in lines if line.strip().startswith(('#', '//', '/*', '*', '--'))]
            
            analysis += f"Total lines: {len(lines)}\n"
            analysis += f"Code lines: {len(non_empty_lines)}\n"
            analysis += f"Comment lines: {len(comment_lines)}\n\n"
            
            # Add the actual code
            analysis += "--- SOURCE CODE ---\n"
            analysis += code_content
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error processing code file: {e}")
            return f"Error reading code file: {str(e)}"
    
    async def _process_structured_data(self, file_path: Path, data_type: str) -> str:
        """Process structured data files (JSON, XML, YAML, TOML)."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            analysis = f"--- {data_type.upper()} FILE: {file_path.name} ---\n\n"
            
            if data_type == 'json':
                try:
                    data = json.loads(content)
                    analysis += f"JSON structure analysis:\n"
                    analysis += f"Type: {type(data).__name__}\n"
                    if isinstance(data, dict):
                        analysis += f"Keys: {list(data.keys())}\n"
                    elif isinstance(data, list):
                        analysis += f"Array length: {len(data)}\n"
                    analysis += "\n"
                except:
                    analysis += "Invalid JSON format\n\n"
            
            elif data_type == 'xml':
                try:
                    root = ET.fromstring(content)
                    analysis += f"XML root element: {root.tag}\n"
                    children = [child.tag for child in root]
                    if children:
                        analysis += f"Child elements: {children}\n"
                    analysis += "\n"
                except:
                    analysis += "Invalid XML format\n\n"
            
            # Add the raw content
            analysis += "--- RAW CONTENT ---\n"
            analysis += content
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error processing structured data: {e}")
            return f"Error reading {data_type} file: {str(e)}"
    
    async def _process_image(self, file_path: Path) -> str:
        """Process image files and extract metadata."""
        try:
            # For now, return image path - actual image processing would return image embeddings
            image = Image.open(file_path)
            
            analysis = f"--- IMAGE FILE: {file_path.name} ---\n\n"
            analysis += f"Format: {image.format}\n"
            analysis += f"Mode: {image.mode}\n"
            analysis += f"Size: {image.size[0]}x{image.size[1]} pixels\n"
            
            # Extract EXIF data if available
            if hasattr(image, '_getexif') and image._getexif():
                analysis += f"EXIF data available: Yes\n"
            else:
                analysis += f"EXIF data available: No\n"
            
            analysis += f"\nImage path for processing: {file_path}\n"
            
            return str(file_path)  # Return path for embedding generation
            
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            return f"Error reading image: {str(e)}"
    
    async def _process_audio(self, file_path: Path) -> str:
        """Process audio files."""
        try:
            # Load audio for analysis
            audio, sr = await asyncio.to_thread(librosa.load, str(file_path))
            duration = len(audio) / sr
            
            analysis = f"--- AUDIO FILE: {file_path.name} ---\n\n"
            analysis += f"Duration: {duration:.2f} seconds\n"
            analysis += f"Sample rate: {sr} Hz\n"
            analysis += f"Channels: {1 if len(audio.shape) == 1 else audio.shape[1]}\n"
            
            # Transcribe audio using Whisper if available
            try:
                model = whisper.load_model("base")
                result = await asyncio.to_thread(model.transcribe, str(file_path))
                transcription = result["text"]
                analysis += f"\n--- TRANSCRIPTION ---\n{transcription}\n"
            except Exception as whisper_error:
                logger.warning(f"Could not transcribe audio: {whisper_error}")
                analysis += f"\nTranscription: Not available\n"
            
            analysis += f"\nAudio path for processing: {file_path}\n"
            
            return str(file_path)  # Return path for embedding generation
            
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            return f"Error reading audio: {str(e)}"
    
    async def _process_video(self, file_path: Path) -> str:
        """Process video files."""
        try:
            # Load video for analysis
            video = mp.VideoFileClip(str(file_path))
            
            analysis = f"--- VIDEO FILE: {file_path.name} ---\n\n"
            analysis += f"Duration: {video.duration:.2f} seconds\n"
            analysis += f"FPS: {video.fps}\n"
            analysis += f"Resolution: {video.w}x{video.h}\n"
            
            # Extract audio for transcription if available
            if video.audio:
                try:
                    # Extract audio to temporary file
                    temp_audio = f"/tmp/{file_path.stem}_audio.wav"
                    video.audio.write_audiofile(temp_audio, verbose=False, logger=None)
                    
                    # Transcribe audio
                    model = whisper.load_model("base")
                    result = await asyncio.to_thread(model.transcribe, temp_audio)
                    transcription = result["text"]
                    analysis += f"\n--- AUDIO TRANSCRIPTION ---\n{transcription}\n"
                    
                    # Clean up temp file
                    os.remove(temp_audio)
                    
                except Exception as audio_error:
                    logger.warning(f"Could not transcribe video audio: {audio_error}")
                    analysis += f"\nAudio transcription: Not available\n"
            
            video.close()
            analysis += f"\nVideo path for processing: {file_path}\n"
            
            return str(file_path)  # Return path for embedding generation
            
        except Exception as e:
            logger.error(f"Error processing video: {e}")
            return f"Error reading video: {str(e)}"
    
    async def _process_generic_file(self, file_path: Path) -> str:
        """Process unknown file types using textract."""
        try:
            content = await asyncio.to_thread(textract.process, str(file_path))
            return content.decode('utf-8', errors='ignore')
        except Exception as e:
            logger.error(f"Error processing generic file: {e}")
            return f"Error reading file: {str(e)}"
    
    def _split_text_into_chunks(
        self, 
        text: str, 
        chunk_size: int, 
        overlap: int
    ) -> List[str]:
        """Split text into overlapping chunks."""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to break at sentence boundaries
            if end < len(text):
                # Look for sentence endings
                for punct in ['. ', '! ', '? ', '\n\n']:
                    last_punct = text.rfind(punct, start, end)
                    if last_punct > start:
                        end = last_punct + len(punct)
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap
            if start >= len(text):
                break
        
        return chunks
    
    async def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of file."""
        try:
            hasher = hashlib.sha256()
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception as e:
            logger.error(f"Error calculating file hash: {e}")
            return "unknown"
    
    def get_supported_extensions(self) -> Dict[str, str]:
        """Get supported file extensions and their types."""
        return self.supported_extensions.copy()
    
    async def batch_process_directory(
        self, 
        directory_path: Union[str, Path],
        recursive: bool = True,
        file_extensions: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Process all supported files in a directory.
        
        Args:
            directory_path: Path to directory
            recursive: Whether to process subdirectories
            file_extensions: Specific extensions to process (None for all supported)
            
        Returns:
            List of all processed documents
        """
        directory_path = Path(directory_path)
        
        if not directory_path.exists() or not directory_path.is_dir():
            raise ValueError(f"Directory not found: {directory_path}")
        
        all_documents = []
        processed_files = []
        
        # Get file pattern
        pattern = "**/*" if recursive else "*"
        
        for file_path in directory_path.glob(pattern):
            if file_path.is_file():
                extension = file_path.suffix.lower()
                
                # Check if we should process this file
                if file_extensions:
                    if extension not in file_extensions:
                        continue
                elif extension not in self.supported_extensions:
                    continue
                
                try:
                    documents = await self.process_file(file_path)
                    all_documents.extend(documents)
                    processed_files.append(str(file_path))
                    logger.info(f"Processed {file_path}: {len(documents)} chunks")
                    
                except Exception as e:
                    logger.error(f"Failed to process {file_path}: {e}")
        
        logger.info(f"Batch processing complete: {len(processed_files)} files, {len(all_documents)} total chunks")
        
        return all_documents


# Global document processor instance
document_processor = DocumentProcessor()