"""
API Provider Configuration and Management
Supports multiple AI service providers with fallback mechanism
"""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional
import os


class TaskType(Enum):
    """Supported AI task types"""
    STT = "speech_to_text"
    TRANSLATION = "translation"
    TTS = "text_to_speech"
    LIP_SYNC = "lip_sync"
    VIDEO_PROCESSING = "video_processing"


class ProviderType(Enum):
    """Supported API providers"""
    OPENAI = "openai"
    GOOGLE_AI = "google_ai"
    ANTHROPIC = "anthropic"
    AZURE_OPENAI = "azure_openai"
    AWS_BEDROCK = "aws_bedrock"
    HUGGING_FACE = "hugging_face"
    COHERE = "cohere"
    GROQ = "groq"
    MISTRAL = "mistral"
    DEEPSEEK = "deepseek"
    TOGETHER_AI = "together_ai"
    REPLICATE = "replicate"
    OPENROUTER = "openrouter"
    ELEVENLABS = "elevenlabs"
    ASSEMBLYAI = "assemblyai"
    DEEPGRAM = "deepgram"
    DEEPL = "deepl"
    GOOGLE_TRANSLATE = "google_translate"
    MICROSOFT_TRANSLATOR = "microsoft_translator"
    DESCRIPT = "descript"
    RUNWAY = "runway"
    LOCAL = "local"


@dataclass
class ProviderConfig:
    """Configuration for an API provider"""
    provider_type: ProviderType
    api_key: Optional[str]
    api_endpoint: Optional[str]
    model_id: Optional[str]
    is_enabled: bool = True
    priority: int = 0  # Lower number = higher priority
    timeout: int = 30
    max_retries: int = 3
    

@dataclass
class TaskProviderMapping:
    """Maps task types to available providers with fallback chain"""
    task_type: TaskType
    primary_providers: List[ProviderConfig]
    fallback_providers: List[ProviderConfig]


class APIProviderManager:
    """Manages multiple API providers with fallback mechanism"""
    
    def __init__(self):
        self.providers: Dict[ProviderType, ProviderConfig] = {}
        self.task_mappings: Dict[TaskType, TaskProviderMapping] = {}
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Initialize all configured providers from environment variables"""
        
        # Speech-to-Text Providers
        self.register_provider(ProviderConfig(
            provider_type=ProviderType.ASSEMBLYAI,
            api_key=os.getenv('ASSEMBLYAI_API_KEY'),
            api_endpoint="https://api.assemblyai.com/v2",
            model_id="best",
            is_enabled=bool(os.getenv('ASSEMBLYAI_API_KEY')),
            priority=1
        ))
        
        self.register_provider(ProviderConfig(
            provider_type=ProviderType.DEEPGRAM,
            api_key=os.getenv('DEEPGRAM_API_KEY'),
            api_endpoint="https://api.deepgram.com/v1",
            model_id="nova-2",
            is_enabled=bool(os.getenv('DEEPGRAM_API_KEY')),
            priority=2
        ))
        
        self.register_provider(ProviderConfig(
            provider_type=ProviderType.OPENAI,
            api_key=os.getenv('OPENAI_API_KEY'),
            api_endpoint="https://api.openai.com/v1",
            model_id="whisper-1",
            is_enabled=bool(os.getenv('OPENAI_API_KEY')),
            priority=3
        ))
        
        self.register_provider(ProviderConfig(
            provider_type=ProviderType.LOCAL,
            api_key=None,
            api_endpoint="local",
            model_id="openai/whisper-medium",
            is_enabled=True,
            priority=10  # Lowest priority fallback
        ))
        
        # Translation Providers
        self.register_provider(ProviderConfig(
            provider_type=ProviderType.DEEPL,
            api_key=os.getenv('DEEPL_API_KEY'),
            api_endpoint="https://api-free.deepl.com/v2" if not os.getenv('DEEPL_PRO') else "https://api.deepl.com/v2",
            model_id="v2",
            is_enabled=bool(os.getenv('DEEPL_API_KEY')),
            priority=1
        ))
        
        self.register_provider(ProviderConfig(
            provider_type=ProviderType.GOOGLE_TRANSLATE,
            api_key=os.getenv('GOOGLE_CLOUD_API_KEY'),
            api_endpoint="https://translation.googleapis.com/language/translate/v2",
            model_id="default",
            is_enabled=bool(os.getenv('GOOGLE_CLOUD_API_KEY')),
            priority=2
        ))
        
        self.register_provider(ProviderConfig(
            provider_type=ProviderType.MICROSOFT_TRANSLATOR,
            api_key=os.getenv('AZURE_TRANSLATOR_KEY'),
            api_endpoint="https://api.cognitive.microsofttranslator.com",
            model_id="v3.0",
            is_enabled=bool(os.getenv('AZURE_TRANSLATOR_KEY')),
            priority=3
        ))
        
        self.register_provider(ProviderConfig(
            provider_type=ProviderType.OPENAI,
            api_key=os.getenv('OPENAI_API_KEY'),
            api_endpoint="https://api.openai.com/v1",
            model_id="gpt-3.5-turbo",
            is_enabled=bool(os.getenv('OPENAI_API_KEY')),
            priority=4
        ))
        
        self.register_provider(ProviderConfig(
            provider_type=ProviderType.GOOGLE_AI,
            api_key=os.getenv('GOOGLE_AI_API_KEY'),
            api_endpoint="https://generativelanguage.googleapis.com/v1beta",
            model_id="gemini-pro",
            is_enabled=bool(os.getenv('GOOGLE_AI_API_KEY')),
            priority=5
        ))
        
        self.register_provider(ProviderConfig(
            provider_type=ProviderType.LOCAL,
            api_key=None,
            api_endpoint="local",
            model_id="Helsinki-NLP/opus-mt",
            is_enabled=True,
            priority=10
        ))
        
        # Text-to-Speech Providers
        self.register_provider(ProviderConfig(
            provider_type=ProviderType.ELEVENLABS,
            api_key=os.getenv('ELEVENLABS_API_KEY'),
            api_endpoint="https://api.elevenlabs.io/v1",
            model_id="eleven_monolingual_v1",
            is_enabled=bool(os.getenv('ELEVENLABS_API_KEY')),
            priority=1
        ))
        
        self.register_provider(ProviderConfig(
            provider_type=ProviderType.GOOGLE_AI,
            api_key=os.getenv('GOOGLE_AI_API_KEY'),
            api_endpoint="https://generativelanguage.googleapis.com/v1beta",
            model_id="gemini-pro",
            is_enabled=bool(os.getenv('GOOGLE_AI_API_KEY')),
            priority=2
        ))
        
        self.register_provider(ProviderConfig(
            provider_type=ProviderType.AZURE_OPENAI,
            api_key=os.getenv('AZURE_API_KEY'),
            api_endpoint=os.getenv('AZURE_ENDPOINT'),
            model_id="tts-1",
            is_enabled=bool(os.getenv('AZURE_API_KEY')),
            priority=3
        ))
        
        self.register_provider(ProviderConfig(
            provider_type=ProviderType.LOCAL,
            api_key=None,
            api_endpoint="local",
            model_id="tts_models/en/ljspeech/glow-tts",
            is_enabled=True,
            priority=10
        ))
        
        # Video Processing Providers
        self.register_provider(ProviderConfig(
            provider_type=ProviderType.REPLICATE,
            api_key=os.getenv('REPLICATE_API_KEY'),
            api_endpoint="https://api.replicate.com/v1",
            model_id="wav2lip",
            is_enabled=bool(os.getenv('REPLICATE_API_KEY')),
            priority=1
        ))
        
        self.register_provider(ProviderConfig(
            provider_type=ProviderType.RUNWAY,
            api_key=os.getenv('RUNWAY_API_KEY'),
            api_endpoint="https://api.runwayml.com/v1",
            model_id="gen2",
            is_enabled=bool(os.getenv('RUNWAY_API_KEY')),
            priority=2
        ))
        
        self.register_provider(ProviderConfig(
            provider_type=ProviderType.LOCAL,
            api_key=None,
            api_endpoint="local",
            model_id="wav2lip-gan",
            is_enabled=True,
            priority=10
        ))
        
        # Setup task mappings
        self._setup_task_mappings()
    
    def register_provider(self, config: ProviderConfig):
        """Register an API provider"""
        self.providers[config.provider_type] = config
    
    def _setup_task_mappings(self):
        """Setup which providers handle which tasks"""
        
        # STT task mapping
        self.task_mappings[TaskType.STT] = TaskProviderMapping(
            task_type=TaskType.STT,
            primary_providers=[
                self.providers.get(ProviderType.ASSEMBLYAI),
                self.providers.get(ProviderType.DEEPGRAM),
                self.providers.get(ProviderType.OPENAI),
            ],
            fallback_providers=[
                self.providers.get(ProviderType.LOCAL),
            ]
        )
        
        # Translation task mapping
        self.task_mappings[TaskType.TRANSLATION] = TaskProviderMapping(
            task_type=TaskType.TRANSLATION,
            primary_providers=[
                self.providers.get(ProviderType.DEEPL),
                self.providers.get(ProviderType.GOOGLE_TRANSLATE),
                self.providers.get(ProviderType.MICROSOFT_TRANSLATOR),
            ],
            fallback_providers=[
                self.providers.get(ProviderType.OPENAI),
                self.providers.get(ProviderType.GOOGLE_AI),
                self.providers.get(ProviderType.LOCAL),
            ]
        )
        
        # TTS task mapping
        self.task_mappings[TaskType.TTS] = TaskProviderMapping(
            task_type=TaskType.TTS,
            primary_providers=[
                self.providers.get(ProviderType.ELEVENLABS),
                self.providers.get(ProviderType.GOOGLE_AI),
            ],
            fallback_providers=[
                self.providers.get(ProviderType.AZURE_OPENAI),
                self.providers.get(ProviderType.LOCAL),
            ]
        )
        
        # Lip-sync task mapping
        self.task_mappings[TaskType.LIP_SYNC] = TaskProviderMapping(
            task_type=TaskType.LIP_SYNC,
            primary_providers=[
                self.providers.get(ProviderType.REPLICATE),
                self.providers.get(ProviderType.RUNWAY),
            ],
            fallback_providers=[
                self.providers.get(ProviderType.LOCAL),
            ]
        )
    
    def get_providers_for_task(self, task_type: TaskType) -> List[ProviderConfig]:
        """Get provider chain for a task (primary + fallback)"""
        mapping = self.task_mappings.get(task_type)
        if not mapping:
            return []
        
        # Filter enabled providers
        primary = [p for p in mapping.primary_providers if p and p.is_enabled]
        fallback = [p for p in mapping.fallback_providers if p and p.is_enabled]
        
        # Sort by priority
        all_providers = primary + fallback
        return sorted(all_providers, key=lambda p: p.priority)
    
    def get_provider(self, provider_type: ProviderType) -> Optional[ProviderConfig]:
        """Get a specific provider config"""
        return self.providers.get(provider_type)
