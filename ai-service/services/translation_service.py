"""
Multi-Provider Translation Service
Handles translation requests across multiple API providers
"""

import logging
from typing import Optional
from dataclasses import dataclass
import httpx
import asyncio

from config.api_providers import APIProviderManager, ProviderType, ProviderConfig

logger = logging.getLogger(__name__)


@dataclass
class TranslationResult:
    """Translation result with metadata"""
    translated_text: str
    source_language: str
    target_language: str
    provider: str
    cost: float = 0.0
    timestamp: str = ""


class TranslationService:
    """Translation service supporting multiple providers"""
    
    def __init__(self, provider_manager: APIProviderManager):
        self.provider_manager = provider_manager
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def translate(
        self,
        text: str,
        target_language: str,
        source_language: str = "en",
        provider: Optional[ProviderConfig] = None
    ) -> TranslationResult:
        """
        Translate text to target language using specified provider
        """
        if not provider:
            providers = self.provider_manager.get_providers_for_task(
                from config.api_providers import TaskType
                TaskType.TRANSLATION
            )
            provider = providers[0] if providers else None
        
        if not provider:
            raise ValueError("No translation provider available")
        
        logger.info(f"Translating to {target_language} using {provider.provider_type.value}")
        
        if provider.provider_type == ProviderType.DEEPL:
            return await self._translate_deepl(text, target_language, provider)
        elif provider.provider_type == ProviderType.GOOGLE_TRANSLATE:
            return await self._translate_google(text, target_language, provider)
        elif provider.provider_type == ProviderType.MICROSOFT_TRANSLATOR:
            return await self._translate_azure(text, target_language, provider)
        elif provider.provider_type == ProviderType.OPENAI:
            return await self._translate_openai(text, target_language, provider)
        elif provider.provider_type == ProviderType.GOOGLE_AI:
            return await self._translate_gemini(text, target_language, provider)
        elif provider.provider_type == ProviderType.LOCAL:
            return await self._translate_local(text, target_language, source_language)
        else:
            raise ValueError(f"Unsupported provider: {provider.provider_type}")
    
    async def _translate_deepl(self, text: str, target_lang: str, provider: ProviderConfig) -> TranslationResult:
        """Translate using DeepL API"""
        try:
            headers = {"Authorization": f"DeepL-Auth-Key {provider.api_key}"}
            data = {
                "text": [text],
                "target_lang": target_lang.upper()
            }
            
            response = await self.client.post(
                f"{provider.api_endpoint}/translate",
                json=data,
                headers=headers,
                timeout=provider.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            translated = result["translations"][0]["text"]
            
            return TranslationResult(
                translated_text=translated,
                source_language="en",
                target_language=target_lang,
                provider="deepl",
                cost=0.0
            )
        except Exception as e:
            logger.error(f"DeepL translation failed: {e}")
            raise
    
    async def _translate_google(self, text: str, target_lang: str, provider: ProviderConfig) -> TranslationResult:
        """Translate using Google Translate API"""
        try:
            params = {
                "key": provider.api_key,
                "q": text,
                "target": target_lang
            }
            
            response = await self.client.post(
                provider.api_endpoint,
                params=params,
                timeout=provider.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            translated = result["data"]["translations"][0]["translatedText"]
            
            return TranslationResult(
                translated_text=translated,
                source_language="en",
                target_language=target_lang,
                provider="google_translate",
                cost=0.0
            )
        except Exception as e:
            logger.error(f"Google Translate failed: {e}")
            raise
    
    async def _translate_azure(self, text: str, target_lang: str, provider: ProviderConfig) -> TranslationResult:
        """Translate using Azure Microsoft Translator API"""
        try:
            headers = {"Ocp-Apim-Subscription-Key": provider.api_key}
            params = {
                "api-version": "3.0",
                "from": "en",
                "to": target_lang
            }
            
            response = await self.client.post(
                f"{provider.api_endpoint}/translate",
                params=params,
                json=[{"Text": text}],
                headers=headers,
                timeout=provider.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            translated = result[0]["translations"][0]["text"]
            
            return TranslationResult(
                translated_text=translated,
                source_language="en",
                target_language=target_lang,
                provider="azure_translator",
                cost=0.0
            )
        except Exception as e:
            logger.error(f"Azure Translator failed: {e}")
            raise
    
    async def _translate_openai(self, text: str, target_lang: str, provider: ProviderConfig) -> TranslationResult:
        """Translate using OpenAI GPT"""
        try:
            import openai
            openai.api_key = provider.api_key
            
            response = await asyncio.to_thread(
                openai.ChatCompletion.create,
                model=provider.model_id,
                messages=[
                    {"role": "system", "content": f"You are a translator. Translate the following text to {target_lang}. Return only the translated text."},
                    {"role": "user", "content": text}
                ]
            )
            
            translated = response.choices[0].message.content.strip()
            
            return TranslationResult(
                translated_text=translated,
                source_language="en",
                target_language=target_lang,
                provider="openai",
                cost=0.0
            )
        except Exception as e:
            logger.error(f"OpenAI translation failed: {e}")
            raise
    
    async def _translate_gemini(self, text: str, target_lang: str, provider: ProviderConfig) -> TranslationResult:
        """Translate using Google Gemini"""
        try:
            import google.generativeai as genai
            genai.configure(api_key=provider.api_key)
            model = genai.GenerativeModel(provider.model_id)
            
            response = await asyncio.to_thread(
                model.generate_content,
                f"Translate the following text to {target_lang}. Return only the translated text.\n\n{text}"
            )
            
            translated = response.text.strip()
            
            return TranslationResult(
                translated_text=translated,
                source_language="en",
                target_language=target_lang,
                provider="gemini",
                cost=0.0
            )
        except Exception as e:
            logger.error(f"Gemini translation failed: {e}")
            raise
    
    async def _translate_local(self, text: str, target_lang: str, source_lang: str = "en") -> TranslationResult:
        """Translate using local Helsinki-NLP model"""
        try:
            from transformers import pipeline
            
            # Map language codes to Helsinki-NLP model format
            model_name = f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"
            translator = pipeline("translation", model=model_name)
            
            result = await asyncio.to_thread(
                translator,
                text,
                max_length=512
            )
            
            translated = result[0]["translation_text"]
            
            return TranslationResult(
                translated_text=translated,
                source_language=source_lang,
                target_language=target_lang,
                provider="local",
                cost=0.0
            )
        except Exception as e:
            logger.error(f"Local translation failed: {e}")
            raise
