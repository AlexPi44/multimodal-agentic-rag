"""Main Reflex application for the Multimodal Agentic RAG System."""

import reflex as rx
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
import os
from pathlib import Path

# Import our core components
from src.core.config import settings
from src.core.logging import get_logger
from src.agents.orchestrator import orchestrator, AgentType
from src.multimodal.processors import document_processor

logger = get_logger(__name__)


class State(rx.State):
    """The app state for the RAG system."""
    
    # Chat state
    messages: List[Dict[str, str]] = []
    current_query: str = ""
    is_loading: bool = False
    
    # File upload state
    uploaded_files: List[str] = []
    processing_status: str = ""
    
    # System state
    system_info: Dict[str, Any] = {}
    agent_type: str = "synthesizer"
    search_mode: str = "hybrid"
    
    # Statistics
    total_documents: int = 0
    processing_time: float = 0.0
    
    async def get_system_capabilities(self):
        """Get system capabilities and statistics."""
        try:
            capabilities = await orchestrator.get_capabilities()
            self.system_info = capabilities
            
            # Get document count from vector store
            from src.core.vector_store import vector_store
            info = await vector_store.get_collection_info()
            self.total_documents = info.get('points_count', 0)
            
        except Exception as e:
            logger.error(f"Error getting system capabilities: {e}")
            self.system_info = {"error": str(e)}
    
    async def send_message(self):
        """Send a message to the RAG system."""
        if not self.current_query.strip():
            return
        
        # Add user message
        user_message = {
            "role": "user",
            "content": self.current_query,
            "timestamp": datetime.now().strftime("%H:%M:%S")
        }
        self.messages.append(user_message)
        
        # Set loading state
        self.is_loading = True
        query = self.current_query
        self.current_query = ""
        
        try:
            start_time = datetime.now()
            
            # Choose between simple and complex query processing
            if self.agent_type == "simple":
                result = await orchestrator.simple_query(
                    query=query,
                    modality="text",
                    top_k=5
                )
                response = result.get("response", "No response generated")
            else:
                # Use plan and execute for complex queries
                agent_enum = AgentType(self.agent_type)
                result = await orchestrator.plan_and_execute(
                    query=query,
                    agent_type=agent_enum
                )
                response = result.get("response", "No response generated")
            
            self.processing_time = (datetime.now() - start_time).total_seconds()
            
            # Add assistant response
            assistant_message = {
                "role": "assistant", 
                "content": response,
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "processing_time": f"{self.processing_time:.2f}s"
            }
            self.messages.append(assistant_message)
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            error_message = {
                "role": "assistant",
                "content": f"Error processing your query: {str(e)}",
                "timestamp": datetime.now().strftime("%H:%M:%S")
            }
            self.messages.append(error_message)
        
        finally:
            self.is_loading = False
    
    async def handle_upload(self, files: List[rx.UploadFile]):
        """Handle file uploads."""
        if not files:
            return
        
        self.processing_status = "Processing uploaded files..."
        
        try:
            for file in files:
                # Save uploaded file
                upload_dir = Path(settings.upload_dir)
                upload_dir.mkdir(exist_ok=True)
                
                file_path = upload_dir / file.filename
                with open(file_path, "wb") as f:
                    f.write(await file.read())
                
                # Process the file
                documents = await document_processor.process_file(file_path)
                
                # Generate embeddings and store
                from src.multimodal.embeddings import embedding_manager
                from src.core.vector_store import vector_store
                
                embeddings = {}
                for modality in ['text', 'image', 'audio']:
                    modal_docs = [doc for doc in documents if doc.get('modality') == modality]
                    if modal_docs:
                        if modality == 'text':
                            emb = await embedding_manager.encode_text([doc['content'] for doc in modal_docs])
                            embeddings[modality] = emb['dense'] if isinstance(emb, dict) else emb
                        elif modality == 'image':
                            emb = await embedding_manager.encode_image([doc['content'] for doc in modal_docs])
                            embeddings[modality] = emb
                        elif modality == 'audio':
                            emb = await embedding_manager.encode_audio([doc['content'] for doc in modal_docs])
                            embeddings[modality] = emb
                
                # Store in vector database
                if embeddings:
                    doc_ids = await vector_store.add_documents(documents, embeddings)
                    self.uploaded_files.append(f"{file.filename} ({len(documents)} chunks)")
                    logger.info(f"Processed and stored {file.filename}: {len(documents)} chunks")
                
            self.processing_status = f"Successfully processed {len(files)} files"
            
            # Update document count
            await self.get_system_capabilities()
            
        except Exception as e:
            logger.error(f"Error processing uploads: {e}")
            self.processing_status = f"Error processing files: {str(e)}"
    
    def clear_messages(self):
        """Clear all chat messages."""
        self.messages = []
    
    def set_agent_type(self, agent_type: str):
        """Set the agent type for processing."""
        self.agent_type = agent_type
    
    def set_search_mode(self, search_mode: str):
        """Set the search mode."""
        self.search_mode = search_mode


def chat_interface() -> rx.Component:
    """Create the chat interface component."""
    return rx.vstack(
        # Chat header
        rx.hstack(
            rx.heading("🤖 Multimodal Agentic RAG", size="lg"),
            rx.spacer(),
            rx.badge(f"Documents: {State.total_documents}", color_scheme="blue"),
            rx.badge(f"Agent: {State.agent_type}", color_scheme="green"),
            width="100%",
            padding="1rem",
            border_bottom="1px solid #e2e8f0"
        ),
        
        # Messages area
        rx.box(
            rx.foreach(
                State.messages,
                lambda message: rx.vstack(
                    rx.hstack(
                        rx.avatar(
                            name="User" if message["role"] == "user" else "Assistant",
                            size="sm"
                        ),
                        rx.vstack(
                            rx.text(
                                message["content"],
                                font_size="sm",
                                line_height="1.5"
                            ),
                            rx.text(
                                f"{message['timestamp']}" + 
                                (f" • {message['processing_time']}" if "processing_time" in message else ""),
                                font_size="xs",
                                color="gray.500"
                            ),
                            align_items="start",
                            spacing="0.25rem"
                        ),
                        align_items="start",
                        spacing="0.75rem",
                        width="100%"
                    ),
                    rx.divider(),
                    width="100%",
                    spacing="0.5rem"
                )
            ),
            height="500px",
            overflow_y="auto",
            padding="1rem",
            width="100%",
            border="1px solid #e2e8f0",
            border_radius="md"
        ),
        
        # Input area
        rx.hstack(
            rx.input(
                placeholder="Ask anything about your documents...",
                value=State.current_query,
                on_change=State.set_current_query,
                width="100%",
                disabled=State.is_loading
            ),
            rx.button(
                "Send" if not State.is_loading else "Processing...",
                on_click=State.send_message,
                disabled=State.is_loading,
                loading=State.is_loading,
                color_scheme="blue"
            ),
            width="100%",
            spacing="0.5rem"
        ),
        
        # Controls
        rx.hstack(
            rx.select(
                ["simple", "researcher", "analyzer", "synthesizer"],
                value=State.agent_type,
                on_change=State.set_agent_type,
                placeholder="Select Agent Type"
            ),
            rx.select(
                ["hybrid", "cosine", "euclidean", "angular", "dot_product"],
                value=State.search_mode,
                on_change=State.set_search_mode,
                placeholder="Search Mode"
            ),
            rx.button(
                "Clear Chat",
                on_click=State.clear_messages,
                variant="outline",
                size="sm"
            ),
            width="100%",
            spacing="0.5rem",
            justify="start"
        ),
        
        width="100%",
        spacing="1rem"
    )


def file_upload_interface() -> rx.Component:
    """Create the file upload interface."""
    return rx.vstack(
        rx.heading("📁 Document Upload", size="md"),
        
        # Upload area
        rx.upload(
            rx.vstack(
                rx.button(
                    "Select Files",
                    color_scheme="blue",
                    border="2px dashed #cbd5e0",
                    padding="2rem",
                    width="100%"
                ),
                rx.text(
                    "Supported: PDF, DOCX, TXT, Code files, Images, Audio, Video",
                    font_size="sm",
                    color="gray.500"
                )
            ),
            accept={
                "application/pdf": [".pdf"],
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document": [".docx"],
                "text/plain": [".txt"],
                "text/markdown": [".md"],
                "text/html": [".html"],
                "application/json": [".json"],
                "text/x-python": [".py"],
                "text/javascript": [".js"],
                "text/css": [".css"],
                "image/*": [".jpg", ".jpeg", ".png", ".gif"],
                "audio/*": [".mp3", ".wav"],
                "video/*": [".mp4", ".avi"]
            },
            multiple=True,
            max_files=10,
            max_size="100MB",
            border="2px dashed #cbd5e0",
            padding="1rem"
        ),
        
        # Upload button
        rx.button(
            "Upload & Process",
            on_click=lambda: State.handle_upload(rx.get_uploaded_files()),
            color_scheme="green",
            width="100%"
        ),
        
        # Status
        rx.cond(
            State.processing_status,
            rx.alert(
                rx.alert_icon(),
                rx.alert_title(State.processing_status),
                status="info"
            )
        ),
        
        # Uploaded files list
        rx.cond(
            State.uploaded_files,
            rx.vstack(
                rx.heading("Uploaded Files", size="sm"),
                rx.foreach(
                    State.uploaded_files,
                    lambda file: rx.text(f"✓ {file}", font_size="sm")
                ),
                width="100%"
            )
        ),
        
        width="100%",
        spacing="1rem"
    )


def system_stats() -> rx.Component:
    """Create system statistics component."""
    return rx.vstack(
        rx.heading("📊 System Statistics", size="md"),
        
        rx.grid(
            rx.stat(
                rx.stat_label("Total Documents"),
                rx.stat_number(State.total_documents),
                rx.stat_help_text("Chunks in vector database")
            ),
            rx.stat(
                rx.stat_label("Last Processing Time"),
                rx.stat_number(f"{State.processing_time:.2f}s"),
                rx.stat_help_text("Query response time")
            ),
            rx.stat(
                rx.stat_label("Agent Mode"),
                rx.stat_number(State.agent_type),
                rx.stat_help_text("Current reasoning mode")
            ),
            columns=3,
            spacing="1rem",
            width="100%"
        ),
        
        # System capabilities
        rx.cond(
            State.system_info,
            rx.vstack(
                rx.heading("System Capabilities", size="sm"),
                rx.text(f"LLM Providers: {', '.join(State.system_info.get('llm_providers', []))}"),
                rx.text(f"Supported Modalities: {', '.join(State.system_info.get('supported_modalities', []))}"),
                rx.text(f"Similarity Methods: {', '.join(State.system_info.get('similarity_methods', []))}"),
                width="100%",
                spacing="0.25rem"
            )
        ),
        
        width="100%",
        spacing="1rem"
    )


def index() -> rx.Component:
    """Main page component."""
    return rx.container(
        rx.vstack(
            # Header
            rx.hstack(
                rx.heading("🔮 Multimodal Agentic RAG System", size="xl"),
                rx.spacer(),
                rx.link(
                    rx.button("GitHub", variant="outline"),
                    href="https://github.com/yourusername/multimodal-agentic-rag"
                ),
                width="100%",
                padding_bottom="1rem"
            ),
            
            # Main content
            rx.grid(
                # Left column - Chat
                rx.grid_item(
                    chat_interface(),
                    col_span=2
                ),
                
                # Right column - Upload and Stats
                rx.grid_item(
                    rx.vstack(
                        file_upload_interface(),
                        rx.divider(),
                        system_stats(),
                        spacing="1rem",
                        width="100%"
                    ),
                    col_span=1
                ),
                
                template_columns="2fr 1fr",
                gap="2rem",
                width="100%"
            ),
            
            # Footer
            rx.divider(),
            rx.text(
                "Built with Reflex, LangChain, and advanced OSS models",
                font_size="sm",
                color="gray.500",
                text_align="center"
            ),
            
            width="100%",
            spacing="2rem",
            padding="2rem"
        ),
        max_width="100%"
    )


# App configuration
app = rx.App(
    theme=rx.theme(
        appearance="light",
        has_background=True,
        radius="medium",
        accent_color="blue"
    )
)

# Add the page
app.add_page(
    index,
    title="Multimodal Agentic RAG System",
    description="Advanced RAG system with multimodal capabilities and agentic reasoning"
)

# Initialize system on startup
@app.on_load
async def on_load():
    """Initialize system capabilities on app load."""
    state = State()
    await state.get_system_capabilities()


if __name__ == "__main__":
    app.run()