"""Advanced agentic orchestration with LangChain integration."""

import asyncio
import json
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime
from enum import Enum

# LangChain imports
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import BaseTool, Tool
from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.runnables import RunnablePassthrough
from langchain_experimental.plan_and_execute import PlanAndExecute, load_agent_executor, load_chat_planner

# Local imports
from src.core.config import settings
from src.core.logging import get_logger
from src.core.vector_store import vector_store
from src.multimodal.embeddings import embedding_manager
from src.agents.llm_providers import llm_manager

logger = get_logger(__name__)


class AgentType(Enum):
    """Types of agents available."""
    RESEARCHER = "researcher"
    ANALYZER = "analyzer"
    SYNTHESIZER = "synthesizer"
    PLANNER = "planner"
    EXECUTOR = "executor"


class RAGTool(BaseTool):
    """RAG search tool for agents."""
    
    name = "rag_search"
    description = "Search through the knowledge base using semantic similarity"
    
    def __init__(self):
        super().__init__()
        
    async def _arun(
        self,
        query: str,
        modality: str = "text",
        top_k: int = 5,
        score_threshold: float = 0.7
    ) -> str:
        """Async implementation of RAG search."""
        try:
            # Generate query embedding
            if modality == "text":
                query_embeddings = await embedding_manager.encode_text(query)
                if isinstance(query_embeddings, dict):
                    query_embedding = query_embeddings.get('dense')
                else:
                    query_embedding = query_embeddings
            else:
                # For multimodal queries, we'd handle differently
                query_embedding = await embedding_manager.encode_text(query)
                if isinstance(query_embedding, dict):
                    query_embedding = query_embedding.get('dense')
            
            # Search vector store
            results = await vector_store.similarity_search(
                query_embedding=query_embedding[0] if len(query_embedding.shape) > 1 else query_embedding,
                modality=modality,
                limit=top_k,
                score_threshold=score_threshold
            )
            
            if not results:
                return "No relevant information found in the knowledge base."
            
            # Format results
            formatted_results = []
            for i, result in enumerate(results):
                formatted_results.append(
                    f"Result {i+1} (Score: {result['score']:.3f}):\n"
                    f"Content: {result['content'][:200]}...\n"
                    f"Source: {result.get('filename', 'Unknown')}\n"
                )
            
            return "\n".join(formatted_results)
            
        except Exception as e:
            logger.error(f"RAG search error: {e}")
            return f"Error during search: {str(e)}"
    
    def _run(self, query: str, **kwargs) -> str:
        """Sync implementation (fallback)."""
        return asyncio.run(self._arun(query, **kwargs))


class MultimodalSearchTool(BaseTool):
    """Multimodal search tool that can handle text, images, and audio."""
    
    name = "multimodal_search"
    description = "Search using multiple modalities (text, image, audio) with hybrid similarity"
    
    async def _arun(
        self,
        text_query: Optional[str] = None,
        image_path: Optional[str] = None,
        audio_path: Optional[str] = None,
        modality_weights: Optional[Dict[str, float]] = None
    ) -> str:
        """Perform multimodal search."""
        try:
            query_embeddings = {}
            
            # Generate embeddings for each provided modality
            if text_query:
                text_emb = await embedding_manager.encode_text(text_query)
                if isinstance(text_emb, dict):
                    query_embeddings['text'] = text_emb['dense'][0]
                else:
                    query_embeddings['text'] = text_emb[0]
                    
            if image_path:
                image_emb = await embedding_manager.encode_image(image_path)
                query_embeddings['image'] = image_emb[0]
                
            if audio_path:
                audio_emb = await embedding_manager.encode_audio(audio_path)
                query_embeddings['audio'] = audio_emb[0]
            
            if not query_embeddings:
                return "No valid query inputs provided."
            
            # Perform hybrid search
            results = await vector_store.hybrid_search(
                query_embeddings=query_embeddings,
                modality_weights=modality_weights,
                limit=5,
                score_threshold=0.6
            )
            
            if not results:
                return "No relevant multimodal information found."
            
            # Format results
            formatted_results = []
            for i, result in enumerate(results):
                formatted_results.append(
                    f"Result {i+1} (Combined Score: {result['combined_score']:.3f}):\n"
                    f"Matched Modalities: {', '.join(result['matched_modalities'])}\n"
                    f"Content: {result['content'][:150]}...\n"
                    f"Source: {result.get('filename', 'Unknown')}\n"
                )
            
            return "\n".join(formatted_results)
            
        except Exception as e:
            logger.error(f"Multimodal search error: {e}")
            return f"Error during multimodal search: {str(e)}"
    
    def _run(self, **kwargs) -> str:
        """Sync implementation (fallback)."""
        return asyncio.run(self._arun(**kwargs))


class DocumentProcessorTool(BaseTool):
    """Tool for processing and indexing new documents."""
    
    name = "document_processor"
    description = "Process and index new documents into the knowledge base"
    
    async def _arun(
        self,
        file_path: str,
        document_type: str = "auto"
    ) -> str:
        """Process and index a document."""
        try:
            from src.multimodal.processors import DocumentProcessor
            processor = DocumentProcessor()
            
            # Process document
            processed_docs = await processor.process_file(file_path, document_type)
            
            # Generate embeddings
            embeddings = {}
            for modality in ['text', 'image', 'audio']:
                if any(doc.get('modality') == modality for doc in processed_docs):
                    modal_docs = [doc for doc in processed_docs if doc.get('modality') == modality]
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
            doc_ids = await vector_store.add_documents(processed_docs, embeddings)
            
            return f"Successfully processed and indexed {len(processed_docs)} document chunks. Document IDs: {doc_ids[:3]}..."
            
        except Exception as e:
            logger.error(f"Document processing error: {e}")
            return f"Error processing document: {str(e)}"
    
    def _run(self, file_path: str, document_type: str = "auto") -> str:
        """Sync implementation (fallback)."""
        return asyncio.run(self._arun(file_path, document_type))


class AnalyticsTool(BaseTool):
    """Tool for analyzing knowledge base statistics and insights."""
    
    name = "analytics"
    description = "Analyze knowledge base statistics, document distribution, and search patterns"
    
    async def _arun(self, analysis_type: str = "overview") -> str:
        """Perform analytics on the knowledge base."""
        try:
            if analysis_type == "overview":
                # Get collection info
                info = await vector_store.get_collection_info()
                
                analysis = f"""Knowledge Base Analytics Overview:
                
Collection Name: {info.get('name', 'Unknown')}
Total Documents: {info.get('points_count', 0)}
Vector Dimensions: {info.get('vector_size', 'N/A')}
Index Status: {info.get('status', 'Unknown')}
Segments: {info.get('segments_count', 0)}

Embedding Models:
- Text: {embedding_manager.text_embedder.model_name}
- Image: {embedding_manager.image_embedder.model_name}
- Audio: {embedding_manager.audio_embedder.model_name}

Search Capabilities:
- Similarity Methods: cosine, euclidean, dot_product, angular, hybrid
- Multimodal Support: Yes
- Real-time Processing: Yes
"""
                return analysis
                
            elif analysis_type == "embedding_dimensions":
                dims = embedding_manager.get_embedding_dimensions()
                return f"Embedding Dimensions: {dims}"
                
            else:
                return f"Unknown analysis type: {analysis_type}"
                
        except Exception as e:
            logger.error(f"Analytics error: {e}")
            return f"Error during analytics: {str(e)}"
    
    def _run(self, analysis_type: str = "overview") -> str:
        """Sync implementation (fallback)."""
        return asyncio.run(self._arun(analysis_type))


class AgentOrchestrator:
    """Advanced agent orchestrator with planning and execution capabilities."""
    
    def __init__(self):
        """Initialize the agent orchestrator."""
        self.llm_manager = llm_manager
        self.tools = self._create_tools()
        self.agents = {}
        self.memory = ConversationBufferWindowMemory(
            k=10,
            memory_key="chat_history",
            return_messages=True
        )
        self._initialize_agents()
        
        logger.info("Initialized agent orchestrator")
    
    def _create_tools(self) -> List[BaseTool]:
        """Create tools for agents."""
        return [
            RAGTool(),
            MultimodalSearchTool(),
            DocumentProcessorTool(),
            AnalyticsTool(),
        ]
    
    def _initialize_agents(self):
        """Initialize different types of agents."""
        try:
            # Get default LLM
            default_provider = self.llm_manager.get_provider()
            
            # Create prompts for different agent types
            self.prompts = {
                AgentType.RESEARCHER: ChatPromptTemplate.from_messages([
                    ("system", """You are a research agent specializing in information discovery.
                    Your role is to search through knowledge bases, find relevant information,
                    and provide comprehensive research summaries. Use the available tools to
                    gather information from multiple sources and modalities."""),
                    MessagesPlaceholder(variable_name="chat_history"),
                    ("human", "{input}"),
                    MessagesPlaceholder(variable_name="agent_scratchpad"),
                ]),
                
                AgentType.ANALYZER: ChatPromptTemplate.from_messages([
                    ("system", """You are an analytical agent specializing in data analysis.
                    Your role is to analyze information, identify patterns, extract insights,
                    and provide detailed analytical reports. Focus on critical thinking
                    and evidence-based conclusions."""),
                    MessagesPlaceholder(variable_name="chat_history"),
                    ("human", "{input}"),
                    MessagesPlaceholder(variable_name="agent_scratchpad"),
                ]),
                
                AgentType.SYNTHESIZER: ChatPromptTemplate.from_messages([
                    ("system", """You are a synthesis agent specializing in information integration.
                    Your role is to combine information from multiple sources, create coherent
                    summaries, and generate comprehensive responses that address user queries
                    using all available information."""),
                    MessagesPlaceholder(variable_name="chat_history"),
                    ("human", "{input}"),
                    MessagesPlaceholder(variable_name="agent_scratchpad"),
                ]),
            }
            
            # Note: For full agent creation, we'd need proper LangChain LLM integration
            # This is a framework for when the dependencies are properly installed
            
            logger.info(f"Initialized agent prompts for: {list(self.prompts.keys())}")
            
        except Exception as e:
            logger.warning(f"Could not fully initialize agents (likely missing dependencies): {e}")
    
    async def plan_and_execute(
        self,
        query: str,
        agent_type: AgentType = AgentType.SYNTHESIZER,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Plan and execute a complex query using the specified agent type."""
        try:
            start_time = datetime.now()
            
            # Step 1: Planning phase
            plan = await self._create_plan(query, agent_type, context)
            
            # Step 2: Execution phase
            results = await self._execute_plan(plan, agent_type)
            
            # Step 3: Synthesis phase
            final_response = await self._synthesize_results(query, results, agent_type)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "query": query,
                "agent_type": agent_type.value,
                "plan": plan,
                "results": results,
                "response": final_response,
                "execution_time": execution_time,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Plan and execute error: {e}")
            return {
                "query": query,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def _create_plan(
        self,
        query: str,
        agent_type: AgentType,
        context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, str]]:
        """Create an execution plan for the query."""
        try:
            # Simple planning logic - in production, use LangChain's PlanAndExecute
            plan = []
            
            # Analyze query to determine required actions
            if "search" in query.lower() or "find" in query.lower():
                plan.append({
                    "action": "rag_search",
                    "description": "Search knowledge base for relevant information",
                    "params": {"query": query}
                })
            
            if "image" in query.lower() or "picture" in query.lower():
                plan.append({
                    "action": "multimodal_search",
                    "description": "Search using multimodal capabilities",
                    "params": {"text_query": query}
                })
            
            if "analyze" in query.lower() or "statistics" in query.lower():
                plan.append({
                    "action": "analytics",
                    "description": "Perform analytics on knowledge base",
                    "params": {"analysis_type": "overview"}
                })
            
            # Default search if no specific actions identified
            if not plan:
                plan.append({
                    "action": "rag_search",
                    "description": "Perform general knowledge search",
                    "params": {"query": query}
                })
            
            # Add synthesis step
            plan.append({
                "action": "synthesize",
                "description": "Synthesize results into coherent response",
                "params": {"query": query}
            })
            
            logger.info(f"Created plan with {len(plan)} steps")
            return plan
            
        except Exception as e:
            logger.error(f"Planning error: {e}")
            return [{"action": "error", "description": str(e)}]
    
    async def _execute_plan(
        self,
        plan: List[Dict[str, str]],
        agent_type: AgentType
    ) -> List[Dict[str, Any]]:
        """Execute the planned steps."""
        results = []
        
        for step in plan:
            try:
                action = step["action"]
                params = step.get("params", {})
                
                if action == "rag_search":
                    tool = RAGTool()
                    result = await tool._arun(**params)
                elif action == "multimodal_search":
                    tool = MultimodalSearchTool()
                    result = await tool._arun(**params)
                elif action == "analytics":
                    tool = AnalyticsTool()
                    result = await tool._arun(**params)
                elif action == "synthesize":
                    # Synthesis happens in the next phase
                    result = "Ready for synthesis"
                else:
                    result = f"Unknown action: {action}"
                
                results.append({
                    "step": step,
                    "result": result,
                    "status": "success"
                })
                
            except Exception as e:
                logger.error(f"Step execution error: {e}")
                results.append({
                    "step": step,
                    "result": str(e),
                    "status": "error"
                })
        
        return results
    
    async def _synthesize_results(
        self,
        query: str,
        results: List[Dict[str, Any]],
        agent_type: AgentType
    ) -> str:
        """Synthesize results into a final response."""
        try:
            # Collect all successful results
            successful_results = [r["result"] for r in results if r["status"] == "success"]
            
            if not successful_results:
                return "I apologize, but I couldn't find relevant information to answer your query."
            
            # Create context for LLM
            context = "\n\n".join([f"Source {i+1}: {result}" for i, result in enumerate(successful_results)])
            
            # Create synthesis prompt
            synthesis_prompt = f"""Based on the following information retrieved from the knowledge base, 
            please provide a comprehensive and accurate answer to the user's query.

            User Query: {query}

            Retrieved Information:
            {context}

            Please synthesize this information into a clear, helpful response that directly addresses the user's question.
            If the information is insufficient, please state what additional information would be helpful."""
            
            # Use LLM to generate response
            messages = [
                SystemMessage(content="You are a helpful AI assistant that synthesizes information to answer user queries."),
                HumanMessage(content=synthesis_prompt)
            ]
            
            response = await self.llm_manager.generate_response(messages, stream=False)
            
            return response if isinstance(response, str) else "Error generating response"
            
        except Exception as e:
            logger.error(f"Synthesis error: {e}")
            return f"Error synthesizing results: {str(e)}"
    
    async def simple_query(
        self,
        query: str,
        modality: str = "text",
        top_k: int = 5
    ) -> Dict[str, Any]:
        """Handle simple queries without full planning."""
        try:
            start_time = datetime.now()
            
            # Perform RAG search
            tool = RAGTool()
            search_results = await tool._arun(query, modality=modality, top_k=top_k)
            
            # Generate response using LLM
            messages = [
                SystemMessage(content="You are a helpful AI assistant. Use the provided context to answer the user's question accurately."),
                HumanMessage(content=f"Context: {search_results}\n\nQuestion: {query}")
            ]
            
            response = await self.llm_manager.generate_response(messages)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "query": query,
                "context": search_results,
                "response": response,
                "execution_time": execution_time,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Simple query error: {e}")
            return {
                "query": query,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def get_capabilities(self) -> Dict[str, Any]:
        """Get information about orchestrator capabilities."""
        return {
            "agent_types": [agent.value for agent in AgentType],
            "tools": [tool.name for tool in self.tools],
            "llm_providers": self.llm_manager.list_providers(),
            "embedding_models": embedding_manager.get_embedding_dimensions(),
            "supported_modalities": ["text", "image", "audio"],
            "similarity_methods": ["cosine", "euclidean", "dot_product", "angular", "hybrid"]
        }


# Global orchestrator instance
orchestrator = AgentOrchestrator()