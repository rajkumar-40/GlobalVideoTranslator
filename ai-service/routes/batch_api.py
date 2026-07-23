"""
Batch Video Translation API Endpoints
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional
import logging

from agents.batch_video_agent import BatchVideoAgent, BatchTranslationJob
from config.api_providers import APIProviderManager

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/batch", tags=["batch"])

# Global instances
provider_manager = APIProviderManager()
agent = BatchVideoAgent(provider_manager)


class CreateBatchJobRequest(BaseModel):
    """Request to create a batch job"""
    user_id: str
    video_paths: List[str]
    target_languages: List[str]
    job_name: Optional[str] = None
    concurrent_tasks: int = 2
    priority: int = 5


class BatchJobResponse(BaseModel):
    """Response with batch job information"""
    job_id: str
    user_id: str
    name: str
    total_videos: int
    target_languages: List[str]
    overall_status: str
    progress: float
    completed_videos: int
    failed_videos: int
    total_api_cost: float
    api_fallback_count: int


@router.post("/jobs/create", response_model=BatchJobResponse)
async def create_batch_job(request: CreateBatchJobRequest) -> BatchJobResponse:
    """
    Create a new batch video translation job
    
    Example:
    ```json
    {
        "user_id": "user123",
        "video_paths": ["/videos/video1.mp4", "/videos/video2.mp4"],
        "target_languages": ["es", "fr", "de"],
        "job_name": "Spanish-French-German Translation",
        "concurrent_tasks": 2,
        "priority": 5
    }
    ```
    """
    try:
        job = await agent.create_batch_job(
            user_id=request.user_id,
            video_paths=request.video_paths,
            target_languages=request.target_languages,
            job_name=request.job_name,
            concurrent_tasks=request.concurrent_tasks,
            priority=request.priority
        )
        
        return BatchJobResponse(
            job_id=job.job_id,
            user_id=job.user_id,
            name=job.name,
            total_videos=job.total_videos,
            target_languages=job.target_languages,
            overall_status=job.overall_status.value,
            progress=job.progress,
            completed_videos=job.completed_videos,
            failed_videos=job.failed_videos,
            total_api_cost=job.total_api_cost,
            api_fallback_count=job.api_fallback_count
        )
    except Exception as e:
        logger.error(f"Failed to create batch job: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/jobs/{job_id}/process")
async def process_batch_job(job_id: str, background_tasks: BackgroundTasks) -> dict:
    """
    Start processing a batch job
    
    Processing happens in background. Use /jobs/{job_id}/status to monitor progress.
    """
    try:
        job = agent.get_job_status(job_id)
        if not job:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        
        # Add processing to background tasks
        background_tasks.add_task(agent.process_batch_job, job_id)
        
        return {
            "job_id": job_id,
            "status": "processing_started",
            "message": "Batch job processing started in background"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start batch job processing: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/jobs/{job_id}/status", response_model=BatchJobResponse)
async def get_job_status(job_id: str) -> BatchJobResponse:
    """
    Get current status of a batch job
    """
    try:
        job = agent.get_job_status(job_id)
        if not job:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        
        return BatchJobResponse(
            job_id=job.job_id,
            user_id=job.user_id,
            name=job.name,
            total_videos=job.total_videos,
            target_languages=job.target_languages,
            overall_status=job.overall_status.value,
            progress=job.progress,
            completed_videos=job.completed_videos,
            failed_videos=job.failed_videos,
            total_api_cost=job.total_api_cost,
            api_fallback_count=job.api_fallback_count
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get job status: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/jobs")
async def list_active_jobs() -> dict:
    """
    List all active batch jobs
    """
    try:
        jobs = agent.list_active_jobs()
        return {
            "total_jobs": len(jobs),
            "jobs": [
                {
                    "job_id": job.job_id,
                    "user_id": job.user_id,
                    "name": job.name,
                    "progress": job.progress,
                    "overall_status": job.overall_status.value
                }
                for job in jobs
            ]
        }
    except Exception as e:
        logger.error(f"Failed to list jobs: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/jobs/{job_id}/details")
async def get_job_details(job_id: str) -> dict:
    """
    Get detailed information about a batch job including individual task statuses
    """
    try:
        job = agent.get_job_status(job_id)
        if not job:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        
        tasks = []
        for task in job.tasks:
            tasks.append({
                "task_id": task.task_id,
                "video_path": task.video_path,
                "status": task.status.value,
                "progress": task.progress,
                "output_paths": task.output_paths,
                "api_calls": task.api_calls,
                "total_api_cost": task.total_api_cost,
                "error": task.error
            })
        
        return {
            "job_id": job.job_id,
            "user_id": job.user_id,
            "name": job.name,
            "overall_status": job.overall_status.value,
            "progress": job.progress,
            "total_videos": job.total_videos,
            "completed_videos": job.completed_videos,
            "failed_videos": job.failed_videos,
            "total_api_cost": job.total_api_cost,
            "api_fallback_count": job.api_fallback_count,
            "tasks": tasks
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get job details: {e}")
        raise HTTPException(status_code=400, detail=str(e))
