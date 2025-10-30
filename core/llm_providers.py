"""Advanced LLM providers with support for OSS models like GPT-OSS20B."""

import asyncio
import json
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from abc import ABC, abstractmethod

# LangChain imports
from langchain.llms.base import LLM
from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_community.llms import Ollama

# OSS model support
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig,
    TextIteratorStreamer
)
from threading import Thread
import requests

from src.core.config import settings
from src.core.logging import get_logger

logger = get_logger(__name__)


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    async def generate(
        self, 
        messages: List[BaseMessage], 
        temperature: float = 0.7,
        max_tokens: int = 4096,
        stream: bool = False
    ) -> Union[str, AsyncGenerator[str, None]]:
        """Generate response from messages."""
        pass

    @abstractmethod
    async def embed_query(self, text: str) -> List[float]:
        """Generate embeddings for a query."""
        pass


class OpenAIProvider(BaseLLMProvider):
    """OpenAI LLM provider with GPT models."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4-turbo-preview"):
        """Initialize OpenAI provider."""
        self.api_key = api_key or settings.openai_api_key
        self.model = model
        self.client = ChatOpenAI(
            api_key=self.api_key,
            model=self.model,
            temperature=settings.default_temperature,
            max_tokens=settings.max_tokens
        )
        logger.info(f"Initialized OpenAI provider with model: {model}")

    async def generate(
        self, 
        messages: List[BaseMessage], 
        temperature: float = 0.7,
        max_tokens: int = 4096,
        stream: bool = False
    ) -> Union[str, AsyncGenerator[str, None]]:
        """Generate response using OpenAI."""
        try:
            if stream:
                async def stream_generator():
                    async for chunk in self.client.astream(
                        messages, 
                        temperature=temperature,
                        max_tokens=max_tokens
                    ):
                        if chunk.content:
                            yield chunk.content
                return stream_generator()
            else:
                response = await self.client.ainvoke(
                    messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                return response.content
        except Exception as e:
            logger.error(f"OpenAI generation error: {e}")
            raise

    async def embed_query(self, text: str) -> List[float]:
        """Generate embeddings using OpenAI."""
        try:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key=self.api_key)
            
            response = await client.embeddings.create(
                input=text,
                model="text-embedding-3-large"
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"OpenAI embedding error: {e}")
            raise


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude provider."""

    def __init__(self, api_key: Optional[str] = None, model: str = "claude-3-opus-20240229"):
        """Initialize Anthropic provider."""
        self.api_key = api_key or settings.anthropic_api_key
        self.model = model
        self.client = ChatAnthropic(
            api_key=self.api_key,
            model=self.model,
            temperature=settings.default_temperature,
            max_tokens=settings.max_tokens
        )
        logger.info(f"Initialized Anthropic provider with model: {model}")

    async def generate(
        self, 
        messages: List[BaseMessage], 
        temperature: float = 0.7,
        max_tokens: int = 4096,
        stream: bool = False
    ) -> Union[str, AsyncGenerator[str, None]]:
        """Generate response using Anthropic."""
        try:
            if stream:
                async def stream_generator():
                    async for chunk in self.client.astream(
                        messages,
                        temperature=temperature,
                        max_tokens=max_tokens
                    ):
                        if chunk.content:
                            yield chunk.content
                return stream_generator()
            else:
                response = await self.client.ainvoke(
                    messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                return response.content
        except Exception as e:
            logger.error(f"Anthropic generation error: {e}")
            raise

    async def embed_query(self, text: str) -> List[float]:
        """Anthropic doesn't provide embeddings, fallback to sentence transformers."""
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('all-MiniLM-L6-v2')
            embedding = await asyncio.to_thread(model.encode, text)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Fallback embedding error: {e}")
            raise


class OllamaProvider(BaseLLMProvider):
    """Ollama provider for local OSS models."""

    def __init__(self, model: str = "llama2", host: str = "http://localhost:11434"):
        """Initialize Ollama provider."""
        self.model = model
        self.host = host
        self.client = Ollama(
            model=self.model,
            base_url=self.host,
            temperature=settings.default_temperature
        )
        logger.info(f"Initialized Ollama provider with model: {model}")

    async def generate(
        self, 
        messages: List[BaseMessage], 
        temperature: float = 0.7,
        max_tokens: int = 4096,
        stream: bool = False
    ) -> Union[str, AsyncGenerator[str, None]]:
        """Generate response using Ollama."""
        try:
            # Convert messages to prompt
            prompt = self._messages_to_prompt(messages)
            
            if stream:
                async def stream_generator():
                    async for chunk in self.client.astream(
                        prompt,
                        temperature=temperature,
                        max_tokens=max_tokens
                    ):
                        yield chunk
                return stream_generator()
            else:
                response = await self.client.ainvoke(
                    prompt,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                return response
        except Exception as e:
            logger.error(f"Ollama generation error: {e}")
            raise

    def _messages_to_prompt(self, messages: List[BaseMessage]) -> str:
        """Convert messages to prompt format."""
        prompt = ""
        for message in messages:
            if isinstance(message, SystemMessage):
                prompt += f"System: {message.content}\\n"
            elif isinstance(message, HumanMessage):
                prompt += f"Human: {message.content}\\n"
            elif isinstance(message, AIMessage):
                prompt += f"Assistant: {message.content}\\n"
        prompt += "Assistant: "
        return prompt

    async def embed_query(self, text: str) -> List[float]:
        """Generate embeddings using Ollama."""
        try:
            url = f"{self.host}/api/embeddings"
            payload = {"model": self.model, "prompt": text}
            
            async with asyncio.to_thread(requests.post, url, json=payload) as response:
                if response.status_code == 200:
                    return response.json()["embedding"]
                else:
                    raise Exception(f"Ollama embedding failed: {response.status_code}")
        except Exception as e:
            logger.error(f"Ollama embedding error: {e}")
            raise


class GPTOSS20BProvider(BaseLLMProvider):
    """Provider for GPT-OSS 20B model using transformers."""

    def __init__(self, model_name: str = "microsoft/DialoGPT-large", device: str = "auto"):
        """Initialize GPT-OSS 20B provider.
        
        Args:
            model_name: Hugging Face model name
            device: Device to run model on
        """
        self.model_name = model_name
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Configure quantization for large models
        if "20b" in model_name.lower() or "13b" in model_name.lower():
            self.quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        else:
            self.quantization_config = None
        
        self._load_model()
        logger.info(f"Initialized GPT-OSS provider with model: {model_name}")

    def _load_model(self):
        """Load the model and tokenizer."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                padding_side="left"
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            model_kwargs = {}
            if self.quantization_config:
                model_kwargs["quantization_config"] = self.quantization_config
                model_kwargs["device_map"] = "auto"
            else:
                model_kwargs["torch_dtype"] = torch.float16
                model_kwargs["device_map"] = self.device
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                **model_kwargs
            )
            
            if not self.quantization_config:
                self.model = self.model.to(self.device)
                
        except Exception as e:
            logger.error(f"Failed to load GPT-OSS model: {e}")
            raise

    async def generate(
        self, 
        messages: List[BaseMessage], 
        temperature: float = 0.7,
        max_tokens: int = 4096,
        stream: bool = False
    ) -> Union[str, AsyncGenerator[str, None]]:
        """Generate response using GPT-OSS model."""
        try:
            prompt = self._messages_to_prompt(messages)
            
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=2048
            ).to(self.device)
            
            generation_kwargs = {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "do_sample": True,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }
            
            if stream:
                async def stream_generator():
                    streamer = TextIteratorStreamer(
                        self.tokenizer, 
                        skip_prompt=True, 
                        skip_special_tokens=True
                    )
                    generation_kwargs["streamer"] = streamer
                    
                    thread = Thread(
                        target=self.model.generate,
                        args=(inputs.input_ids,),
                        kwargs=generation_kwargs
                    )
                    thread.start()
                    
                    for token in streamer:
                        yield token
                    
                    thread.join()
                    
                return stream_generator()
            else:
                with torch.no_grad():
                    outputs = await asyncio.to_thread(
                        self.model.generate,
                        inputs.input_ids,
                        **generation_kwargs
                    )
                
                response = self.tokenizer.decode(
                    outputs[0][inputs.input_ids.shape[1]:], 
                    skip_special_tokens=True
                )
                return response.strip()
                
        except Exception as e:
            logger.error(f"GPT-OSS generation error: {e}")
            raise

    def _messages_to_prompt(self, messages: List[BaseMessage]) -> str:
        """Convert messages to prompt format."""
        prompt = ""
        for message in messages:
            if isinstance(message, SystemMessage):
                prompt += f"<|system|>\\n{message.content}\\n"
            elif isinstance(message, HumanMessage):
                prompt += f"<|user|>\\n{message.content}\\n"
            elif isinstance(message, AIMessage):
                prompt += f"<|assistant|>\\n{message.content}\\n"
        prompt += "<|assistant|>\\n"
        return prompt

    async def embed_query(self, text: str) -> List[float]:
        """Generate embeddings using the model's hidden states."""
        try:
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                outputs = await asyncio.to_thread(
                    self.model,
                    **inputs,
                    output_hidden_states=True
                )
                
            # Use mean pooling of last hidden state
            hidden_states = outputs.hidden_states[-1]
            embeddings = torch.mean(hidden_states, dim=1).squeeze()
            
            return embeddings.cpu().numpy().tolist()
            
        except Exception as e:
            logger.error(f"GPT-OSS embedding error: {e}")
            raise


class LLMProviderManager:
    """Manager for all LLM providers."""

    def __init__(self):
        """Initialize the LLM provider manager."""
        self.providers = {}
        self._initialize_providers()
        
    def _initialize_providers(self):
        """Initialize available providers based on configuration."""
        try:
            # OpenAI
            if settings.openai_api_key:
                self.providers["openai"] = OpenAIProvider(
                    api_key=settings.openai_api_key,
                    model=settings.default_llm_model if settings.default_llm_provider == "openai" else "gpt-4-turbo-preview"
                )
                
            # Anthropic
            if settings.anthropic_api_key:
                self.providers["anthropic"] = AnthropicProvider(
                    api_key=settings.anthropic_api_key,
                    model=settings.default_llm_model if settings.default_llm_provider == "anthropic" else "claude-3-opus-20240229"
                )
            
            # Ollama (local models)
            if settings.use_local_models:
                self.providers["ollama"] = OllamaProvider(
                    model=settings.default_llm_model if settings.default_llm_provider == "ollama" else "llama2",
                    host=settings.ollama_host
                )
                
            # GPT-OSS 20B
            if settings.use_local_models:
                try:
                    self.providers["gpt-oss"] = GPTOSS20BProvider(
                        model_name="microsoft/DialoGPT-large"  # Replace with actual OSS 20B model
                    )
                except Exception as e:
                    logger.warning(f"Failed to initialize GPT-OSS provider: {e}")
            
            logger.info(f"Initialized LLM providers: {list(self.providers.keys())}")
            
        except Exception as e:
            logger.error(f"Failed to initialize providers: {e}")
            raise

    def get_provider(self, provider_name: Optional[str] = None) -> BaseLLMProvider:
        """Get a specific provider or the default one.
        
        Args:
            provider_name: Name of the provider to get
            
        Returns:
            LLM provider instance
        """
        if provider_name is None:
            provider_name = settings.default_llm_provider
            
        if provider_name not in self.providers:
            available = list(self.providers.keys())
            raise ValueError(f"Provider {provider_name} not available. Available: {available}")
            
        return self.providers[provider_name]

    async def generate_response(
        self,
        messages: List[BaseMessage],
        provider_name: Optional[str] = None,
        temperature: float = None,
        max_tokens: int = None,
        stream: bool = False
    ) -> Union[str, AsyncGenerator[str, None]]:
        """Generate response using specified provider.
        
        Args:
            messages: List of messages for conversation
            provider_name: Provider to use for generation
            temperature: Generation temperature
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response
            
        Returns:
            Generated response or stream
        """
        provider = self.get_provider(provider_name)
        
        temperature = temperature or settings.default_temperature
        max_tokens = max_tokens or settings.max_tokens
        
        return await provider.generate(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream
        )

    def list_providers(self) -> List[str]:
        """List available providers."""
        return list(self.providers.keys())


# Global LLM provider manager
llm_manager = LLMProviderManager()