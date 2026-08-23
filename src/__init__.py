"""
GameMaster AI - Gaming LLM Wrapper Package
==========================================

A specialized LLM wrapper for gaming assistance using open-source models.
Provides builds, strategies, lore, patch analysis, coaching, and more.

Modules:
- model_manager: Handles model loading, quantization, and inference
- prompts: Gaming-specific prompt templates
- gaming_llm: Main wrapper class integrating all features
- knowledge_base: RAG system for gaming knowledge
- cli: Command-line interface
"""

from .model_manager import ModelManager, create_model_manager
from .prompts import (
    GamingContext,
    PromptType,
    GameGenre,
    get_prompt,
    get_available_prompt_types,
    get_supported_games,
    detect_game_genre,
    BASE_SYSTEM_PROMPT
)
from .gaming_llm import GamingLLM, GamingResponse, create_gaming_llm
from .knowledge_base import GamingKnowledgeBase, GamingDocument, create_knowledge_base, create_sample_knowledge_base

__version__ = "1.0.0"
__author__ = "GameMaster AI Team"

__all__ = [
    # Model Manager
    "ModelManager",
    "create_model_manager",
    
    # Prompts
    "GamingContext",
    "PromptType",
    "GameGenre",
    "get_prompt",
    "get_available_prompt_types",
    "get_supported_games",
    "detect_game_genre",
    "BASE_SYSTEM_PROMPT",
    
    # Gaming LLM
    "GamingLLM",
    "GamingResponse",
    "create_gaming_llm",
    
    # Knowledge Base
    "GamingKnowledgeBase",
    "GamingDocument",
    "create_knowledge_base",
    "create_sample_knowledge_base",
]