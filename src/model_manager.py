"""
Model Manager for GameMaster AI
Handles loading, quantization, and management of open-source LLMs.
"""

import os
import torch
from typing import Optional, Dict, Any, List
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Pipeline,
    pipeline
)
from accelerate import infer_auto_device_map, dispatch_model
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages LLM model loading and inference for gaming applications."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the model manager.
        
        Args:
            config: Configuration dictionary with model settings
        """
        self.config = config
        self.model_config = config.get("model", {})
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.device = self._get_device()
        self._model_loaded = False
        
    def _get_device(self) -> str:
        """Determine the best device for model inference."""
        device_config = self.model_config.get("device", "auto")
        
        if device_config == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device_config
    
    def _get_quantization_config(self) -> Optional[BitsAndBytesConfig]:
        """Create quantization configuration based on settings."""
        quantization = self.model_config.get("quantization", "none").lower()
        
        if quantization == "4bit":
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif quantization == "8bit":
            return BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False
            )
        return None
    
    def load_model(self) -> bool:
        """
        Load the model and tokenizer.
        
        Returns:
            bool: True if model loaded successfully
        """
        try:
            model_id = self.model_config.get("model_id", "meta-llama/Meta-Llama-3-8B-Instruct")
            logger.info(f"Loading model: {model_id}")
            logger.info(f"Device: {self.device}")
            logger.info(f"Quantization: {self.model_config.get('quantization', 'none')}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                trust_remote_code=True,
                padding_side="left"
            )
            
            # Set pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with quantization
            quantization_config = self._get_quantization_config()
            
            model_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch.float16 if self.device != "cpu" else torch.float32,
                "device_map": "auto" if self.device == "cuda" else None,
            }
            
            if quantization_config:
                model_kwargs["quantization_config"] = quantization_config
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                **model_kwargs
            )
            
            # Create pipeline for easier inference
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1,
                torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
            )
            
            self._model_loaded = True
            logger.info("Model loaded successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        **kwargs
    ) -> str:
        """
        Generate text using the loaded model.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
            top_k: Top-k sampling
            repetition_penalty: Repetition penalty
            **kwargs: Additional generation arguments
            
        Returns:
            str: Generated text
        """
        if not self._model_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Use config defaults if not provided
        gen_config = {
            "max_new_tokens": max_new_tokens or self.model_config.get("max_new_tokens", 2048),
            "temperature": temperature or self.model_config.get("temperature", 0.7),
            "top_p": top_p or self.model_config.get("top_p", 0.9),
            "top_k": top_k or self.model_config.get("top_k", 50),
            "repetition_penalty": repetition_penalty or self.model_config.get("repetition_penalty", 1.1),
            "do_sample": True,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            **kwargs
        }
        
        # Generate
        with torch.no_grad():
            outputs = self.pipeline(prompt, **gen_config)
        
        # Extract generated text (remove the prompt)
        generated_text = outputs[0]["generated_text"]
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()
        
        return generated_text
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        **generation_kwargs
    ) -> str:
        """
        Chat with the model using conversation format.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            **generation_kwargs: Generation parameters
            
        Returns:
            str: Model response
        """
        if not self._model_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Apply chat template
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        return self.generate(prompt, **generation_kwargs)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if not self._model_loaded:
            return {"status": "not_loaded"}
        
        return {
            "status": "loaded",
            "model_id": self.model_config.get("model_id"),
            "device": self.device,
            "quantization": self.model_config.get("quantization", "none"),
            "max_context_length": self.model_config.get("max_context_length", 8192),
            "model_dtype": str(next(self.model.parameters()).dtype) if self.model else None,
        }
    
    def unload_model(self):
        """Unload model to free memory."""
        if self.model:
            del self.model
            self.model = None
        if self.tokenizer:
            del self.tokenizer
            self.tokenizer = None
        if self.pipeline:
            del self.pipeline
            self.pipeline = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self._model_loaded = False
        logger.info("Model unloaded successfully")


def create_model_manager(config: Dict[str, Any]) -> ModelManager:
    """Factory function to create a ModelManager instance."""
    return ModelManager(config)