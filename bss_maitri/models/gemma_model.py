"""
Gemma 3 model integration for BSS Maitri AI Assistant
"""

import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig,
    pipeline
)
from typing import List, Dict, Any, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class GemmaModel:
    """Gemma 3 model wrapper for BSS Maitri psychological support"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_name = config.get('name', 'google/gemma-2-2b-it')
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.precision = config.get('precision', 'fp16')
        self.max_length = config.get('max_length', 2048)
        
        self.tokenizer = None
        self.model = None
        self.generator = None
        
        self._load_model()
    
    def _get_quantization_config(self) -> Optional[BitsAndBytesConfig]:
        """Configure model quantization for efficient inference"""
        if self.precision == 'fp16' and self.device == 'cuda':
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        return None
    
    def _load_model(self):
        """Load Gemma model and tokenizer"""
        try:
            logger.info(f"Loading Gemma model: {self.model_name}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # Configure model loading
            model_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch.float16 if self.precision == 'fp16' else torch.float32,
            }
            
            # Add quantization if specified
            quantization_config = self._get_quantization_config()
            if quantization_config:
                model_kwargs["quantization_config"] = quantization_config
                model_kwargs["device_map"] = "auto"
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs
            )
            
            # Move to device if not using device_map
            if "device_map" not in model_kwargs:
                self.model = self.model.to(self.device)
            
            # Create generation pipeline
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == 'cuda' else -1,
                torch_dtype=torch.float16 if self.precision == 'fp16' else torch.float32
            )
            
            logger.info("Gemma model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading Gemma model: {e}")
            raise
    
    def generate_response(
        self, 
        prompt: str, 
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs
    ) -> str:
        """Generate response using Gemma model"""
        try:
            # Format prompt for psychological support context
            formatted_prompt = self._format_psychological_prompt(prompt)
            
            # Generate response
            result = self.generator(
                formatted_prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                **kwargs
            )
            
            # Extract generated text
            generated_text = result[0]['generated_text']
            response = generated_text[len(formatted_prompt):].strip()
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I'm sorry, I'm having trouble processing your request right now. How are you feeling?"
    
    def _format_psychological_prompt(self, user_input: str) -> str:
        """Format prompt for psychological support context"""
        system_prompt = """You are Maitri, an AI assistant aboard the Bharatiya Space Station. You are designed to provide psychological support and companionship to crew members. Your responses should be:

1. Empathetic and understanding
2. Professionally supportive but warm
3. Brief and relevant to space station operations
4. Focused on crew mental health and well-being
5. Culturally sensitive to Indian values and traditions

Keep responses concise but meaningful. If you detect serious emotional distress, acknowledge it and suggest speaking with ground control or medical team.

User: {user_input}
Maitri:"""
        
        return system_prompt.format(user_input=user_input)
    
    def generate_intervention(self, emotion_state: Dict[str, float], context: str = "") -> str:
        """Generate targeted psychological intervention based on detected emotions"""
        
        # Determine primary emotion
        primary_emotion = max(emotion_state.items(), key=lambda x: x[1])
        emotion_name, confidence = primary_emotion
        
        if confidence < 0.5:
            emotion_name = "neutral"
        
        # Create intervention prompt
        intervention_prompts = {
            "stress": "The crew member seems stressed. Provide a brief, calming response with a practical stress management technique suitable for space station environment.",
            "anxiety": "The crew member appears anxious. Offer reassurance and a simple breathing or mindfulness technique.",
            "sadness": "The crew member seems sad or low. Provide gentle support and encouragement while maintaining professional boundaries.",
            "anger": "The crew member appears frustrated or angry. Offer understanding and suggest constructive ways to manage these feelings.",
            "fear": "The crew member seems fearful. Provide reassurance while acknowledging their concerns as valid.",
            "neutral": "Engage in a supportive check-in conversation to maintain positive crew morale."
        }
        
        prompt = intervention_prompts.get(emotion_name, intervention_prompts["neutral"])
        if context:
            prompt += f" Context: {context}"
        
        return self.generate_response(prompt, max_new_tokens=150, temperature=0.6)
    
    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, 'model') and self.model:
            del self.model
        if hasattr(self, 'tokenizer') and self.tokenizer:
            del self.tokenizer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None