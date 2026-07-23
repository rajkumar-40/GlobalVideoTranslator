"""
Batch Video Translation Agent
Orchestrates multi-language video translation using multi-API fallback system
"""

import asyncio
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid

from config.api_providers import APIProviderManager, TaskType, ProviderType, ProviderConfig
from services.translation_service import TranslationService
from services.speech_service import SpeechService
from services.video_processor import VideoProcessor
from database.models import JobStatus, BatchJob, VideoTask


logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Status of individual video task"""
    PENDING = "pending"
    EXTRACTING_AUDIO = "extracting_audio"
    TRANSCRIBING = "transcribing"
    TRANSLATING = "translating"
    GENERATING_SPEECH = "generating_speech"
    SYNCING_LIP = "syncing_lip"
    FINALIZING = "finalizing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class VideoTranslationTask:
    """Represents a single video translation task"""
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    video_path: str = ""
    target_languages: List[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    progress: float = 0.0
    error: Optional[str] = None
    output_paths: Dict[str, str] = field(default_factory=dict)  # language -> output_path
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Intermediate results
    extracted_audio: Optional[str] = None
    transcribed_text: Optional[str] = None
    translations: Dict[str, str] = field(default_factory=dict)  # language -> translated_text
    generated_audio: Dict[str, str] = field(default_factory=dict)  # language -> audio_path
    
    # API usage tracking
    api_calls: List[Dict] = field(default_factory=list)
    total_api_cost: float = 0.0


@dataclass
class BatchTranslationJob:
    """Batch job containing multiple video translation tasks"""
    job_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    name: str = ""
    description: str = ""
    video_paths: List[str] = field(default_factory=list)
    target_languages: List[str] = field(default_factory=list)
    priority: int = 5  # 1-10, higher = more important
    
    tasks: List[VideoTranslationTask] = field(default_factory=list)
    overall_status: JobStatus = JobStatus.PENDING
    progress: float = 0.0
    
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Batch settings
    concurrent_tasks: int = 2
    retry_failed: bool = True
    max_retries: int = 3
    timeout_per_video: int = 3600  # seconds
    
    # Statistics
    total_videos: int = 0
    completed_videos: int = 0
    failed_videos: int = 0
    total_api_cost: float = 0.0
    total_duration: int = 0  # seconds
    api_fallback_count: int = 0


class BatchVideoAgent:
    """Agent for batch video translation with multi-API fallback"""
    
    def __init__(self, provider_manager: APIProviderManager):
        self.provider_manager = provider_manager
        self.translation_service = TranslationService(provider_manager)
        self.speech_service = SpeechService(provider_manager)
        self.video_processor = VideoProcessor()
        self.active_jobs: Dict[str, BatchTranslationJob] = {}
    
    async def create_batch_job(
        self,
        user_id: str,
        video_paths: List[str],
        target_languages: List[str],
        job_name: str = "",
        concurrent_tasks: int = 2,
        priority: int = 5
    ) -> BatchTranslationJob:
        """
        Create a new batch translation job
        
        Args:
            user_id: User identifier
            video_paths: List of video file paths to process
            target_languages: List of target language codes (e.g., ['es', 'fr', 'de'])
            job_name: Human-readable job name
            concurrent_tasks: Number of videos to process concurrently
            priority: Job priority (1-10)
        
        Returns:
            Created batch job
        """
        job = BatchTranslationJob(
            user_id=user_id,
            name=job_name or f"Batch Translation {datetime.utcnow().isoformat()}",
            video_paths=video_paths,
            target_languages=target_languages,
            priority=priority,
            concurrent_tasks=concurrent_tasks,
            total_videos=len(video_paths)
        )
        
        # Create individual tasks
        for video_path in video_paths:
            task = VideoTranslationTask(
                video_path=video_path,
                target_languages=target_languages
            )
            job.tasks.append(task)
        
        self.active_jobs[job.job_id] = job
        logger.info(f"Created batch job {job.job_id} with {len(video_paths)} videos")
        
        return job
    
    async def process_batch_job(self, job_id: str) -> BatchTranslationJob:
        """
        Process an entire batch job with concurrent task execution
        
        Args:
            job_id: Batch job identifier
        
        Returns:
            Completed batch job
        """
        job = self.active_jobs.get(job_id)
        if not job:
            raise ValueError(f"Batch job {job_id} not found")
        
        job.overall_status = JobStatus.PROCESSING
        job.started_at = datetime.utcnow()
        
        logger.info(f"Starting batch job {job_id} with {len(job.tasks)} tasks")
        
        try:
            # Process tasks with concurrency control
            semaphore = asyncio.Semaphore(job.concurrent_tasks)
            
            async def process_task_with_semaphore(task: VideoTranslationTask):
                async with semaphore:
                    return await self._process_single_task(job, task)
            
            # Sort tasks by priority (though all same priority initially)
            tasks_to_process = sorted(job.tasks, key=lambda t: job.priority, reverse=True)
            
            # Execute all tasks concurrently
            results = await asyncio.gather(
                *[process_task_with_semaphore(task) for task in tasks_to_process],
                return_exceptions=True
            )
            
            # Update job statistics
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Task processing error: {result}")
                    job.failed_videos += 1
                else:
                    job.completed_videos += 1
                    job.total_api_cost += result.get('api_cost', 0)
            
            job.progress = (job.completed_videos + job.failed_videos) / len(job.tasks)
            job.overall_status = JobStatus.COMPLETED if job.failed_videos == 0 else JobStatus.PARTIAL
            
        except Exception as e:
            logger.error(f"Batch job {job_id} failed: {e}")
            job.overall_status = JobStatus.FAILED
            raise
        finally:
            job.completed_at = datetime.utcnow()
            job.total_duration = int((job.completed_at - job.started_at).total_seconds())
        
        return job
    
    async def _process_single_task(
        self,
        job: BatchTranslationJob,
        task: VideoTranslationTask
    ) -> Dict:
        """
        Process a single video translation task with multi-API fallback
        
        Args:
            job: Parent batch job
            task: Video translation task to process
        
        Returns:
            Task result dictionary with metadata
        """
        task.status = TaskStatus.EXTRACTING_AUDIO
        task.started_at = datetime.utcnow()
        task_result = {'api_cost': 0.0}
        
        try:
            logger.info(f"Processing task {task.task_id}: {task.video_path}")
            
            # Step 1: Extract audio from video
            logger.info(f"[{task.task_id}] Extracting audio...")
            task.extracted_audio = await self.video_processor.extract_audio(task.video_path)
            task.progress = 0.15
            
            # Step 2: Transcribe audio (with multi-API fallback)
            logger.info(f"[{task.task_id}] Transcribing audio...")
            task.status = TaskStatus.TRANSCRIBING
            task.transcribed_text = await self._transcribe_with_fallback(
                task, task.extracted_audio
            )
            task.progress = 0.30
            task_result['api_cost'] += task.api_calls[-1]['cost'] if task.api_calls else 0
            
            # Step 3: Translate to each target language (with multi-API fallback)
            logger.info(f"[{task.task_id}] Translating to {len(task.target_languages)} languages...")
            task.status = TaskStatus.TRANSLATING
            for idx, lang in enumerate(task.target_languages):
                translated = await self._translate_with_fallback(
                    task, task.transcribed_text, lang
                )
                task.translations[lang] = translated
                task.progress = 0.30 + (0.20 * (idx + 1) / len(task.target_languages))
                task_result['api_cost'] += task.api_calls[-1]['cost'] if task.api_calls else 0
            
            # Step 4: Generate speech for each language (with multi-API fallback)
            logger.info(f"[{task.task_id}] Generating speech...")
            task.status = TaskStatus.GENERATING_SPEECH
            for idx, lang in enumerate(task.target_languages):
                audio_path = await self._generate_speech_with_fallback(
                    task, task.translations[lang], lang
                )
                task.generated_audio[lang] = audio_path
                task.progress = 0.50 + (0.25 * (idx + 1) / len(task.target_languages))
                task_result['api_cost'] += task.api_calls[-1]['cost'] if task.api_calls else 0
            
            # Step 5: Lip-sync and finalize (with multi-API fallback)
            logger.info(f"[{task.task_id}] Performing lip-sync...")
            task.status = TaskStatus.SYNCING_LIP
            for idx, lang in enumerate(task.target_languages):
                output_path = await self._lipsync_with_fallback(
                    task, task.video_path, task.generated_audio[lang], lang
                )
                task.output_paths[lang] = output_path
                task.progress = 0.75 + (0.20 * (idx + 1) / len(task.target_languages))
                task_result['api_cost'] += task.api_calls[-1]['cost'] if task.api_calls else 0
            
            task.status = TaskStatus.COMPLETED
            task.progress = 1.0
            task.completed_at = datetime.utcnow()
            
            logger.info(f"[{task.task_id}] Task completed successfully")
            task_result['status'] = 'success'
            task_result['output_paths'] = task.output_paths
            
        except Exception as e:
            logger.error(f"[{task.task_id}] Task failed: {e}")
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.completed_at = datetime.utcnow()
            task_result['status'] = 'failed'
            task_result['error'] = str(e)
        
        task.total_api_cost = task_result['api_cost']
        return task_result
    
    async def _transcribe_with_fallback(
        self,
        task: VideoTranslationTask,
        audio_path: str
    ) -> str:
        """
        Transcribe audio with automatic API fallback
        """
        providers = self.provider_manager.get_providers_for_task(TaskType.STT)
        last_error = None
        
        for provider in providers:
            try:
                logger.info(f"[{task.task_id}] Attempting transcription with {provider.provider_type.value}")
                result = await self.speech_service.transcribe(
                    audio_path=audio_path,
                    provider=provider
                )
                
                task.api_calls.append({
                    'provider': provider.provider_type.value,
                    'task': 'transcription',
                    'status': 'success',
                    'timestamp': datetime.utcnow().isoformat(),
                    'cost': getattr(result, 'cost', 0)
                })
                
                logger.info(f"[{task.task_id}] Transcription successful with {provider.provider_type.value}")
                return result.text
            
            except Exception as e:
                last_error = e
                logger.warning(f"[{task.task_id}] Transcription failed with {provider.provider_type.value}: {e}")
                task.api_calls.append({
                    'provider': provider.provider_type.value,
                    'task': 'transcription',
                    'status': 'failed',
                    'timestamp': datetime.utcnow().isoformat(),
                    'error': str(e)
                })
                job = next((j for j in self.active_jobs.values() if task in j.tasks), None)
                if job:
                    job.api_fallback_count += 1
        
        raise Exception(f"All transcription providers failed. Last error: {last_error}")
    
    async def _translate_with_fallback(
        self,
        task: VideoTranslationTask,
        text: str,
        target_language: str
    ) -> str:
        """
        Translate text with automatic API fallback
        """
        providers = self.provider_manager.get_providers_for_task(TaskType.TRANSLATION)
        last_error = None
        
        for provider in providers:
            try:
                logger.info(f"[{task.task_id}] Attempting translation to {target_language} with {provider.provider_type.value}")
                result = await self.translation_service.translate(
                    text=text,
                    target_language=target_language,
                    provider=provider
                )
                
                task.api_calls.append({
                    'provider': provider.provider_type.value,
                    'task': f'translation_to_{target_language}',
                    'status': 'success',
                    'timestamp': datetime.utcnow().isoformat(),
                    'cost': getattr(result, 'cost', 0)
                })
                
                logger.info(f"[{task.task_id}] Translation to {target_language} successful with {provider.provider_type.value}")
                return result.translated_text
            
            except Exception as e:
                last_error = e
                logger.warning(f"[{task.task_id}] Translation failed with {provider.provider_type.value}: {e}")
                task.api_calls.append({
                    'provider': provider.provider_type.value,
                    'task': f'translation_to_{target_language}',
                    'status': 'failed',
                    'timestamp': datetime.utcnow().isoformat(),
                    'error': str(e)
                })
                job = next((j for j in self.active_jobs.values() if task in j.tasks), None)
                if job:
                    job.api_fallback_count += 1
        
        raise Exception(f"All translation providers failed for {target_language}. Last error: {last_error}")
    
    async def _generate_speech_with_fallback(
        self,
        task: VideoTranslationTask,
        text: str,
        language: str
    ) -> str:
        """
        Generate speech from text with automatic API fallback
        """
        providers = self.provider_manager.get_providers_for_task(TaskType.TTS)
        last_error = None
        
        for provider in providers:
            try:
                logger.info(f"[{task.task_id}] Attempting speech generation for {language} with {provider.provider_type.value}")
                audio_path = await self.speech_service.synthesize(
                    text=text,
                    language=language,
                    provider=provider
                )
                
                task.api_calls.append({
                    'provider': provider.provider_type.value,
                    'task': f'tts_{language}',
                    'status': 'success',
                    'timestamp': datetime.utcnow().isoformat(),
                    'cost': 0  # Cost would be populated by service
                })
                
                logger.info(f"[{task.task_id}] Speech generated for {language} with {provider.provider_type.value}")
                return audio_path
            
            except Exception as e:
                last_error = e
                logger.warning(f"[{task.task_id}] Speech generation failed with {provider.provider_type.value}: {e}")
                task.api_calls.append({
                    'provider': provider.provider_type.value,
                    'task': f'tts_{language}',
                    'status': 'failed',
                    'timestamp': datetime.utcnow().isoformat(),
                    'error': str(e)
                })
                job = next((j for j in self.active_jobs.values() if task in j.tasks), None)
                if job:
                    job.api_fallback_count += 1
        
        raise Exception(f"All TTS providers failed for {language}. Last error: {last_error}")
    
    async def _lipsync_with_fallback(
        self,
        task: VideoTranslationTask,
        video_path: str,
        audio_path: str,
        language: str
    ) -> str:
        """
        Perform lip-sync with automatic API fallback
        """
        providers = self.provider_manager.get_providers_for_task(TaskType.LIP_SYNC)
        last_error = None
        
        for provider in providers:
            try:
                logger.info(f"[{task.task_id}] Attempting lip-sync for {language} with {provider.provider_type.value}")
                output_path = await self.video_processor.lip_sync(
                    video_path=video_path,
                    audio_path=audio_path,
                    output_language=language,
                    provider=provider
                )
                
                task.api_calls.append({
                    'provider': provider.provider_type.value,
                    'task': f'lipsync_{language}',
                    'status': 'success',
                    'timestamp': datetime.utcnow().isoformat(),
                    'cost': 0
                })
                
                logger.info(f"[{task.task_id}] Lip-sync completed for {language} with {provider.provider_type.value}")
                return output_path
            
            except Exception as e:
                last_error = e
                logger.warning(f"[{task.task_id}] Lip-sync failed with {provider.provider_type.value}: {e}")
                task.api_calls.append({
                    'provider': provider.provider_type.value,
                    'task': f'lipsync_{language}',
                    'status': 'failed',
                    'timestamp': datetime.utcnow().isoformat(),
                    'error': str(e)
                })
                job = next((j for j in self.active_jobs.values() if task in j.tasks), None)
                if job:
                    job.api_fallback_count += 1
        
        raise Exception(f"All lip-sync providers failed for {language}. Last error: {last_error}")
    
    def get_job_status(self, job_id: str) -> Optional[BatchTranslationJob]:
        """Get status of a batch job"""
        return self.active_jobs.get(job_id)
    
    def list_active_jobs(self) -> List[BatchTranslationJob]:
        """List all active batch jobs"""
        return list(self.active_jobs.values())
