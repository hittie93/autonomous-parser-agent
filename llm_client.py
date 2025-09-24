"""
LLM Client Integration Module
Supports multiple LLM providers with a unified interface and API key management.
"""

import os
import logging
from typing import Protocol
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)


class LLMClient(Protocol):
    """Protocol defining the interface for all LLM clients."""

    def generate(self, prompt: str, temperature: float = 0.2) -> str:
        """Generates a response for a given prompt."""
        ...


class GroqLLM:
    """Groq integration for fast, efficient inference."""

    def __init__(self):
        try:
            from groq import Groq
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY not found in .env file.")
            self.client = Groq(api_key=api_key)
        except ImportError:
            raise ImportError("Groq SDK not installed. Please run 'pip install groq'.")

    def generate(self, prompt: str, temperature: float = 0.2) -> str:
        """Generates code using Groq's Mixtral model."""
        try:
            response = self.client.chat.completions.create(
            model="llama-3.1-8b-instant",   # ✅ stable
            messages=[
                {"role": "system", "content": "You are an expert Python developer. Generate clean, working code."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=4000
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Groq API call failed: {e}")
            raise


class GeminiLLM:
    """Google Gemini integration for advanced generative capabilities."""

    def __init__(self):
        try:
            import google.generativeai as genai
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY not found in .env file.")
            genai.configure(api_key=api_key)
            self.client = genai.GenerativeModel('gemini-1.5-flash')
        except ImportError:
            raise ImportError("Gemini SDK not installed. Please run 'pip install google-generativeai'.")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {e}")
            raise

    def generate(self, prompt: str, temperature: float = 0.2) -> str:
        """Generates code using the Gemini 1.5 Flash model."""
        try:
            import google.generativeai as genai
            config = genai.types.GenerationConfig(temperature=temperature, max_output_tokens=4096)
            response = self.client.generate_content(prompt, generation_config=config)
            return response.text
        except Exception as e:
            logger.error(f"Gemini API call failed: {e}")
            raise


class MockLLM:
    """A mock LLM client for offline testing and development."""

    def generate(self, prompt: str, temperature: float = 0.2) -> str:
        """Returns a signal to use the fallback template."""
        logger.info("Using MockLLM to generate a sample parser.")
        return "FALLBACK_TEMPLATE"


def create_llm_client(provider: str) -> LLMClient:
    """
    Factory function to instantiate the correct LLM client.

    Args:
        provider: The LLM provider to use ('groq', 'gemini', or 'mock').

    Returns:
        An instance of the requested LLM client.
    """
    providers = {
        "groq": GroqLLM,
        "gemini": GeminiLLM,
        "mock": MockLLM
    }
    
    client_class = providers.get(provider)
    
    if not client_class:
        logger.warning(f"Unknown provider '{provider}'. Falling back to MockLLM.")
        return MockLLM()

    try:
        return client_class()
    except Exception as e:
        logger.error(f"Failed to initialize {provider} client: {e}")
        logger.info("Falling back to MockLLM due to initialization error.")
        return MockLLM()