"""
Core Gaming LLM Wrapper for GameMaster AI.
Integrates model management, gaming prompts, and specialized features.
"""

import logging
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path
import yaml

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

logger = logging.getLogger(__name__)


@dataclass
class GamingResponse:
    """Structured response from gaming LLM."""
    content: str
    prompt_type: PromptType
    game: str
    context: GamingContext
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "content": self.content,
            "prompt_type": self.prompt_type.value,
            "game": self.game,
            "context": {
                "game": self.context.game,
                "genre": self.context.genre.value,
                "player_rank": self.context.player_rank,
                "role": self.context.role,
                "champion_hero": self.context.champion_hero,
                "patch_version": self.context.patch_version,
                "additional_context": self.context.additional_context
            },
            "metadata": self.metadata
        }


class GamingLLM:
    """
    Main Gaming LLM Wrapper class.
    Provides high-level interface for gaming assistance.
    """
    
    def __init__(self, config_path: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Gaming LLM wrapper.
        
        Args:
            config_path: Path to YAML config file
            config: Configuration dictionary (alternative to config_path)
        """
        # Load configuration
        if config_path:
            self.config = self._load_config(config_path)
        elif config:
            self.config = config
        else:
            self.config = self._default_config()
        
        # Initialize model manager
        self.model_manager = create_model_manager(self.config)
        self._model_loaded = False
        
        # Gaming features config
        self.gaming_config = self.config.get("gaming_features", {})
        self.knowledge_base_config = self.config.get("knowledge_base", {})
        
        # Conversation history
        self.conversation_history: List[Dict[str, str]] = []
        self.max_history = 10
        
        logger.info("GameMaster AI initialized")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _default_config(self) -> Dict[str, Any]:
        """Return default configuration."""
        return {
            "model": {
                "name": "llama-3-8b",
                "model_id": "meta-llama/Meta-Llama-3-8B-Instruct",
                "quantization": "4bit",
                "device": "auto",
                "max_context_length": 8192,
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 50,
                "max_new_tokens": 2048,
                "repetition_penalty": 1.1
            },
            "gaming_features": {
                "supported_games": get_supported_games(),
                "build_optimizer": True,
                "strategy_advisor": True,
                "lore_explorer": True,
                "patch_note_analyzer": True,
                "meta_analyzer": True,
                "coach_mode": True
            },
            "knowledge_base": {
                "enabled": False,
                "data_path": "./data/gaming_knowledge",
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                "top_k": 5,
                "chunk_size": 512,
                "chunk_overlap": 50
            }
        }
    
    def load_model(self) -> bool:
        """
        Load the LLM model.
        
        Returns:
            bool: True if successful
        """
        if self._model_loaded:
            logger.info("Model already loaded")
            return True
        
        success = self.model_manager.load_model()
        if success:
            self._model_loaded = True
            logger.info("GameMaster AI model loaded successfully")
        return success
    
    def unload_model(self):
        """Unload the model to free memory."""
        self.model_manager.unload_model()
        self._model_loaded = False
    
    def is_model_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model_loaded
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return self.model_manager.get_model_info()
    
    def _create_context(
        self,
        game: str,
        prompt_type: PromptType,
        player_rank: Optional[str] = None,
        role: Optional[str] = None,
        champion_hero: Optional[str] = None,
        patch_version: Optional[str] = None,
        additional_context: Optional[str] = None
    ) -> GamingContext:
        """Create a gaming context from parameters."""
        genre = detect_game_genre(game)
        
        return GamingContext(
            game=game,
            genre=genre,
            player_rank=player_rank,
            role=role,
            champion_hero=champion_hero,
            patch_version=patch_version,
            additional_context=additional_context
        )
    
    def ask(
        self,
        game: str,
        question: str,
        prompt_type: PromptType = PromptType.GENERAL_GAMING,
        player_rank: Optional[str] = None,
        role: Optional[str] = None,
        champion_hero: Optional[str] = None,
        patch_version: Optional[str] = None,
        use_history: bool = True,
        **generation_kwargs
    ) -> GamingResponse:
        """
        Ask a gaming question and get a specialized response.
        
        Args:
            game: Name of the game
            question: The question or context
            prompt_type: Type of gaming assistance needed
            player_rank: Player's rank/skill level
            role: Player's role in the game
            champion_hero: Specific champion/hero/character
            patch_version: Game patch version
            use_history: Whether to include conversation history
            **generation_kwargs: Additional generation parameters
            
        Returns:
            GamingResponse with the answer
        """
        if not self._model_loaded:
            self.load_model()
        
        # Create context
        context = self._create_context(
            game=game,
            prompt_type=prompt_type,
            player_rank=player_rank,
            role=role,
            champion_hero=champion_hero,
            patch_version=patch_version,
            additional_context=question
        )
        
        # Get specialized prompt
        prompt = get_prompt(prompt_type, context)
        
        # Add conversation history if enabled
        if use_history and self.conversation_history:
            history_text = "\n".join([
                f"{msg['role']}: {msg['content']}" 
                for msg in self.conversation_history[-self.max_history:]
            ])
            prompt = f"{prompt}\n\nPrevious conversation:\n{history_text}\n\nAssistant:"
        
        # Generate response
        try:
            response_text = self.model_manager.generate(prompt, **generation_kwargs)
            
            # Update conversation history
            self.conversation_history.append({"role": "user", "content": question})
            self.conversation_history.append({"role": "assistant", "content": response_text})
            
            # Trim history
            if len(self.conversation_history) > self.max_history * 2:
                self.conversation_history = self.conversation_history[-self.max_history * 2:]
            
            return GamingResponse(
                content=response_text,
                prompt_type=prompt_type,
                game=game,
                context=context,
                metadata={
                    "model_info": self.get_model_info(),
                    "generation_params": generation_kwargs
                }
            )
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise
    
    # Convenience methods for each prompt type
    def get_build(
        self,
        game: str,
        champion_hero: str,
        role: Optional[str] = None,
        player_rank: Optional[str] = None,
        patch_version: Optional[str] = None,
        additional_context: Optional[str] = None
    ) -> GamingResponse:
        """Get optimized build for a champion/hero."""
        return self.ask(
            game=game,
            question=additional_context or f"Create an optimized build for {champion_hero}",
            prompt_type=PromptType.BUILD_OPTIMIZER,
            player_rank=player_rank,
            role=role,
            champion_hero=champion_hero,
            patch_version=patch_version
        )
    
    def get_strategy(
        self,
        game: str,
        situation: str,
        player_rank: Optional[str] = None,
        role: Optional[str] = None,
        patch_version: Optional[str] = None
    ) -> GamingResponse:
        """Get strategic advice."""
        return self.ask(
            game=game,
            question=situation,
            prompt_type=PromptType.STRATEGY_ADVISOR,
            player_rank=player_rank,
            role=role,
            patch_version=patch_version
        )
    
    def get_lore(
        self,
        game: str,
        topic: str,
        character: Optional[str] = None
    ) -> GamingResponse:
        """Explore game lore."""
        return self.ask(
            game=game,
            question=topic,
            prompt_type=PromptType.LORE_EXPLORER,
            champion_hero=character
        )
    
    def analyze_patch(
        self,
        game: str,
        patch_version: str,
        focus: Optional[str] = None
    ) -> GamingResponse:
        """Analyze a game patch."""
        return self.ask(
            game=game,
            question=focus or f"Analyze patch {patch_version}",
            prompt_type=PromptType.PATCH_ANALYZER,
            patch_version=patch_version
        )
    
    def analyze_meta(
        self,
        game: str,
        patch_version: Optional[str] = None,
        player_rank: Optional[str] = None,
        role: Optional[str] = None
    ) -> GamingResponse:
        """Analyze current meta."""
        return self.ask(
            game=game,
            question="Analyze current meta",
            prompt_type=PromptType.META_ANALYZER,
            player_rank=player_rank,
            role=role,
            patch_version=patch_version
        )
    
    def get_coaching(
        self,
        game: str,
        struggles: str,
        player_rank: str,
        role: str,
        champion_hero: Optional[str] = None
    ) -> GamingResponse:
        """Get personalized coaching."""
        return self.ask(
            game=game,
            question=struggles,
            prompt_type=PromptType.COACH_MODE,
            player_rank=player_rank,
            role=role,
            champion_hero=champion_hero
        )
    
    def get_counters(
        self,
        game: str,
        enemy_champion: str,
        player_rank: Optional[str] = None,
        role: Optional[str] = None,
        patch_version: Optional[str] = None
    ) -> GamingResponse:
        """Get counter-pick advice."""
        return self.ask(
            game=game,
            question=f"Counter picks for {enemy_champion}",
            prompt_type=PromptType.COUNTER_PICK,
            player_rank=player_rank,
            role=role,
            champion_hero=enemy_champion,
            patch_version=patch_version
        )
    
    def analyze_team_comp(
        self,
        game: str,
        current_picks: str,
        player_rank: Optional[str] = None,
        patch_version: Optional[str] = None
    ) -> GamingResponse:
        """Analyze team composition."""
        return self.ask(
            game=game,
            question=current_picks,
            prompt_type=PromptType.TEAM_COMP,
            player_rank=player_rank,
            patch_version=patch_version
        )
    
    def get_itemization(
        self,
        game: str,
        champion_hero: str,
        role: Optional[str] = None,
        player_rank: Optional[str] = None,
        patch_version: Optional[str] = None,
        game_state: Optional[str] = None
    ) -> GamingResponse:
        """Get itemization guide."""
        return self.ask(
            game=game,
            question=game_state or f"Itemization guide for {champion_hero}",
            prompt_type=PromptType.ITEMIZATION,
            player_rank=player_rank,
            role=role,
            champion_hero=champion_hero,
            patch_version=patch_version
        )
    
    def get_mechanics_guide(
        self,
        game: str,
        champion_hero: str,
        role: Optional[str] = None,
        player_rank: Optional[str] = None,
        specific_mechanics: Optional[str] = None
    ) -> GamingResponse:
        """Get mechanics guide."""
        return self.ask(
            game=game,
            question=specific_mechanics or f"Mechanics guide for {champion_hero}",
            prompt_type=PromptType.MECHANICS_GUIDE,
            player_rank=player_rank,
            role=role,
            champion_hero=champion_hero
        )
    
    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []
    
    def get_available_features(self) -> Dict[str, Any]:
        """Get available gaming features and prompt types."""
        return {
            "prompt_types": [pt.value for pt in get_available_prompt_types()],
            "supported_games": get_supported_games(),
            "enabled_features": {
                k: v for k, v in self.gaming_config.items() 
                if isinstance(v, bool) and v
            }
        }


def create_gaming_llm(config_path: Optional[str] = None, config: Optional[Dict[str, Any]] = None) -> GamingLLM:
    """Factory function to create a GamingLLM instance."""
    return GamingLLM(config_path=config_path, config=config)