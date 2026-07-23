"""
Multi-Provider Speech-to-Text and Text-to-Speech Service
"""

import logging
from typing import Optional
from dataclasses import dataclass
import httpx
import asyncio
import base64

from config.api_providers import APIProviderManager, ProviderType, ProviderConfig

logger = logging.getLogger(__name__)


@dataclass
class TranscriptionResult:
    """Transcription result with metadata"""
    text: str
    language: str
    provider: str
    confidence: float = 0.0
    cost: float = 0.0


class SpeechService:
    """Speech processing service supporting multiple providers"""
    
    def __init__(self, provider_manager: APIProviderManager):
        self.provider_manager = provider_manager
        self.client = httpx.AsyncClient(timeout=60.0)
    
    async def transcribe(
        self,
        audio_path: str,
        provider: Optional[ProviderConfig] = None,
        language: Optional[str] = None
    ) -> TranscriptionResult:
        """
        Transcribe audio file to text
        """
        if not provider:
            from config.api_providers import TaskType
            providers = self.provider_manager.get_providers_for_task(TaskType.STT)
            provider = providers[0] if providers else None
        
        if not provider:
            raise ValueError("No speech-to-text provider available")
        
        logger.info(f"Transcribing {audio_path} using {provider.provider_type.value}")
        
        if provider.provider_type == ProviderType.ASSEMBLYAI:
            return await self._transcribe_assemblyai(audio_path, provider)
        elif provider.provider_type == ProviderType.DEEPGRAM:
            return await self._transcribe_deepgram(audio_path, provider)
        elif provider.provider_type == ProviderType.OPENAI:
            return await self._transcribe_openai(audio_path, provider)
        elif provider.provider_type == ProviderType.LOCAL:
            return await self._transcribe_local(audio_path, language)
        else:
            raise ValueError(f"Unsupported STT provider: {provider.provider_type}")
    
    async def synthesize(
        self,
        text: str,
        language: str,
        provider: Optional[ProviderConfig] = None,
        output_path: Optional[str] = None
    ) -> str:
        """
        Synthesize speech from text
        
        Returns:
            Path to generated audio file
        """
        if not provider:
            from config.api_providers import TaskType
            providers = self.provider_manager.get_providers_for_task(TaskType.TTS)
            provider = providers[0] if providers else None
        
        if not provider:
            raise ValueError("No text-to-speech provider available")
        
        logger.info(f"Synthesizing speech using {provider.provider_type.value}")
        
        if provider.provider_type == ProviderType.ELEVENLABS:
            return await self._synthesize_elevenlabs(text, language, provider, output_path)
        elif provider.provider_type == ProviderType.GOOGLE_AI:
            return await self._synthesize_google(text, language, provider, output_path)
        elif provider.provider_type == ProviderType.AZURE_OPENAI:
            return await self._synthesize_azure(text, language, provider, output_path)
        elif provider.provider_type == ProviderType.LOCAL:
            return await self._synthesize_local(text, language, output_path)
        else:
            raise ValueError(f"Unsupported TTS provider: {provider.provider_type}")
    
    async def _transcribe_assemblyai(self, audio_path: str, provider: ProviderConfig) -> TranscriptionResult:
        """Transcribe using AssemblyAI API"""
        try:
            headers = {"Authorization": provider.api_key}
            
            # Read audio file
            with open(audio_path, "rb") as f:
                audio_data = f.read()
            
            # Submit transcription
            response = await self.client.post(
                f"{provider.api_endpoint}/upload",
                content=audio_data,
                headers=headers,
                timeout=provider.timeout
            )
            response.raise_for_status()
            upload_url = response.json()["upload_url"]
            
            # Request transcription
            transcription_response = await self.client.post(
                f"{provider.api_endpoint}/transcript",
                json={"audio_url": upload_url},
                headers=headers,
                timeout=provider.timeout
            )
            transcription_response.raise_for_status()
            transcript_id = transcription_response.json()["id"]
            
            # Poll for completion
            while True:
                result_response = await self.client.get(
                    f"{provider.api_endpoint}/transcript/{transcript_id}",
                    headers=headers,
                    timeout=provider.timeout
                )
                result_response.raise_for_status()
                result = result_response.json()
                
                if result["status"] == "completed":
                    return TranscriptionResult(
                        text=result["text"],
                        language="en",
                        provider="assemblyai",
                        confidence=result.get("confidence", 0.0),
                        cost=0.0
                    )
                elif result["status"] == "failed":
                    raise Exception(f"Transcription failed: {result.get('error')}")
                
                await asyncio.sleep(2)
        except Exception as e:
            logger.error(f"AssemblyAI transcription failed: {e}")
            raise
    
    async def _transcribe_deepgram(self, audio_path: str, provider: ProviderConfig) -> TranscriptionResult:
        """Transcribe using Deepgram API"""
        try:
            headers = {
                "Authorization": f"Token {provider.api_key}",
                "Content-Type": "audio/wav"
            }
            
            with open(audio_path, "rb") as f:
                audio_data = f.read()
            
            response = await self.client.post(
                f"{provider.api_endpoint}/listen",
                content=audio_data,
                headers=headers,
                params={"model": provider.model_id},
                timeout=provider.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            transcript = result["results"]["channels"][0]["alternatives"][0]["transcript"]
            
            return TranscriptionResult(
                text=transcript,
                language="en",
                provider="deepgram",
                confidence=result["results"]["channels"][0]["alternatives"][0].get("confidence", 0.0),
                cost=0.0
            )
        except Exception as e:
            logger.error(f"Deepgram transcription failed: {e}")
            raise
    
    async def _transcribe_openai(self, audio_path: str, provider: ProviderConfig) -> TranscriptionResult:
        """Transcribe using OpenAI Whisper API"""
        try:
            import openai
            openai.api_key = provider.api_key
            
            with open(audio_path, "rb") as f:
                transcript = await asyncio.to_thread(
                    openai.Audio.transcribe,
                    provider.model_id,
                    f
                )
            
            return TranscriptionResult(
                text=transcript["text"],
                language=transcript.get("language", "en"),
                provider="openai_whisper",
                confidence=1.0,
                cost=0.0
            )
        except Exception as e:
            logger.error(f"OpenAI Whisper transcription failed: {e}")
            raise
    
    async def _transcribe_local(self, audio_path: str, language: Optional[str] = None) -> TranscriptionResult:
        """Transcribe using local Whisper model"""
        try:
            import whisper
            
            model = await asyncio.to_thread(whisper.load_model, "base")
            result = await asyncio.to_thread(model.transcribe, audio_path)
            
            return TranscriptionResult(
                text=result["text"],
                language=result.get("language", "en"),
                provider="local_whisper",
                confidence=1.0,
                cost=0.0
            )
        except Exception as e:
            logger.error(f"Local Whisper transcription failed: {e}")
            raise
    
    async def _synthesize_elevenlabs(self, text: str, language: str, provider: ProviderConfig, output_path: Optional[str]) -> str:
        """Synthesize speech using ElevenLabs API"""
        try:
            headers = {"xi-api-key": provider.api_key}
            
            # Use voice ID based on language
            voice_id = self._get_elevenlabs_voice(language)
            
            response = await self.client.post(
                f"{provider.api_endpoint}/text-to-speech/{voice_id}",
                json={"text": text},
                headers=headers,
                timeout=provider.timeout
            )
            response.raise_for_status()
            
            # Save audio file
            if not output_path:
                output_path = f"/tmp/audio_{language}_{hash(text)}.mp3"
            
            with open(output_path, "wb") as f:
                f.write(response.content)
            
            logger.info(f"Speech synthesized and saved to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"ElevenLabs synthesis failed: {e}")
            raise
    
    async def _synthesize_google(self, text: str, language: str, provider: ProviderConfig, output_path: Optional[str]) -> str:
        """Synthesize speech using Google Cloud TTS"""
        try:
            from google.cloud import texttospeech
            
            client = texttospeech.TextToSpeechClient()
            synthesis_input = texttospeech.SynthesisInput(text=text)
            
            voice = texttospeech.VoiceSelectionParams(
                language_code=f"{language}-{language.upper()}"
            )
            
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.MP3
            )
            
            response = client.synthesize_speech(
                input=synthesis_input,
                voice=voice,
                audio_config=audio_config
            )
            
            if not output_path:
                output_path = f"/tmp/audio_{language}_{hash(text)}.mp3"
            
            with open(output_path, "wb") as f:
                f.write(response.audio_content)
            
            logger.info(f"Speech synthesized and saved to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Google TTS synthesis failed: {e}")
            raise
    
    async def _synthesize_azure(self, text: str, language: str, provider: ProviderConfig, output_path: Optional[str]) -> str:
        """Synthesize speech using Azure TTS"""
        try:
            import azure.cognitiveservices.speech as speechsdk
            
            speech_config = speechsdk.SpeechConfig(
                subscription=provider.api_key,
                region=provider.api_endpoint.split(".")[0]
            )
            
            if not output_path:
                output_path = f"/tmp/audio_{language}_{hash(text)}.mp3"
            
            audio_output = speechsdk.audio.AudioOutputConfig(filename=output_path)
            speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_output)
            
            ssml = self._create_ssml(text, language)
            result = speech_synthesizer.speak_ssml_async(ssml).get()
            
            if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                logger.info(f"Speech synthesized and saved to {output_path}")
                return output_path
            else:
                raise Exception(f"Speech synthesis failed: {result.reason}")
        except Exception as e:
            logger.error(f"Azure TTS synthesis failed: {e}")
            raise
    
    async def _synthesize_local(self, text: str, language: str, output_path: Optional[str]) -> str:
        """Synthesize speech using local TTS model"""
        try:
            from TTS.api import TTS
            
            tts = TTS(model_name="tts_models/en/ljspeech/glow-tts", gpu=False)
            
            if not output_path:
                output_path = f"/tmp/audio_{language}_{hash(text)}.wav"
            
            await asyncio.to_thread(tts.tts_to_file, text=text, file_path=output_path)
            
            logger.info(f"Speech synthesized and saved to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Local TTS synthesis failed: {e}")
            raise
    
    def _get_elevenlabs_voice(self, language: str) -> str:
        """Get ElevenLabs voice ID for language"""
        voice_mapping = {
            "en": "21m00Tcm4TlvDq8ikWAM",  # Rachel
            "es": "zcAOhNBS3c14rBihk51B",  # Pablo (Spanish)
            "fr": "VR6AewLTigWG4xSOukaG",  # Cecilia (French)
            "de": "EXAVITQu4vr4xnSDxMaL",  # Gerrit (German)
            "it": "pFZP5JQG7iQjIQuC4Hyc",  # Matteo (Italian)
            "pt": "jBpfuIE2acCO8z5wL7Cc",  # Diniz (Portuguese)
        }
        return voice_mapping.get(language, voice_mapping["en"])
    
    def _create_ssml(self, text: str, language: str) -> str:
        """Create SSML for Azure TTS"""
        lang_map = {
            "en": "en-US",
            "es": "es-ES",
            "fr": "fr-FR",
            "de": "de-DE",
            "it": "it-IT",
            "pt": "pt-BR",
        }
        lang_code = lang_map.get(language, "en-US")
        return f'<speak version="1.0" xml:lang="{lang_code}"><voice name="{lang_code}-Neural"><prosody pitch="0%" rate="0%">{text}</prosody></voice></speak>'
