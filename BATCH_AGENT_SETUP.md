# Batch Video Translation Agent Setup Guide

## Overview

The Batch Video Translation Agent is a comprehensive system for processing multiple videos with automatic language translation and dubbing. It features:

- **Multi-API Fallback System**: Automatically switches between multiple AI service providers when one fails
- **Concurrent Processing**: Process multiple videos simultaneously
- **Multi-Language Support**: Translate and dub videos into multiple target languages in a single job
- **Cost Optimization**: Tracks API usage and selects cost-effective providers
- **Robust Error Handling**: Automatic retry with fallback providers
- **Progress Tracking**: Real-time monitoring of batch and individual task progress

## Supported API Providers

### Speech-to-Text (STT)
1. **AssemblyAI** - High accuracy, diarization support
2. **Deepgram** - Real-time STT, fast processing
3. **OpenAI Whisper** - Reliable, open-source model option
4. **Local Whisper** - Fallback, no API cost

### Translation
1. **DeepL** - Highest quality translations
2. **Google Translate** - Wide language support
3. **Microsoft Translator** - Enterprise support
4. **OpenAI GPT** - Context-aware translations
5. **Google Gemini** - Multimodal capabilities
6. **Local Helsinki-NLP** - Fallback option

### Text-to-Speech (TTS)
1. **ElevenLabs** - Natural voice quality, voice cloning
2. **Google Cloud TTS** - Multiple voices and languages
3. **Azure TTS** - Enterprise-grade synthesis
4. **Local TTS** - Fallback using local models

### Video Processing (Lip-Sync)
1. **Replicate** - Wav2Lip model hosting
2. **Runway** - Video generation and editing
3. **Local Wav2Lip** - Fallback using local model

## Installation

### 1. Environment Setup

```bash
# Copy environment template
cp ai-service/.env.example ai-service/.env

# Edit .env with your API keys
vim ai-service/.env
```

### 2. Install Dependencies

```bash
cd ai-service
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Required System Dependencies

```bash
# For video processing
sudo apt-get install ffmpeg  # Ubuntu/Debian
brew install ffmpeg          # macOS
# Windows: Download from https://ffmpeg.org/download.html
```

## Configuration

### Provider Priority Configuration

Edit `ai-service/config/api_providers.py` to set provider priorities (lower number = higher priority):

```python
self.register_provider(ProviderConfig(
    provider_type=ProviderType.DEEPL,
    api_key=os.getenv('DEEPL_API_KEY'),
    priority=1,  # Highest priority
    is_enabled=True
))
```

## Usage

### 1. Create a Batch Job

```bash
curl -X POST http://localhost:9000/batch/jobs/create \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user123",
    "video_paths": ["/videos/video1.mp4", "/videos/video2.mp4"],
    "target_languages": ["es", "fr", "de"],
    "job_name": "Multi-Language Dubbing",
    "concurrent_tasks": 2,
    "priority": 5
  }'
```

**Response:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "user_id": "user123",
  "name": "Multi-Language Dubbing",
  "total_videos": 2,
  "target_languages": ["es", "fr", "de"],
  "overall_status": "pending",
  "progress": 0.0,
  "completed_videos": 0,
  "failed_videos": 0,
  "total_api_cost": 0.0,
  "api_fallback_count": 0
}
```

### 2. Start Processing

```bash
curl -X POST http://localhost:9000/batch/jobs/{job_id}/process
```

**Response:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "processing_started",
  "message": "Batch job processing started in background"
}
```

### 3. Check Job Status

```bash
curl http://localhost:9000/batch/jobs/{job_id}/status
```

**Response:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "user_id": "user123",
  "name": "Multi-Language Dubbing",
  "total_videos": 2,
  "target_languages": ["es", "fr", "de"],
  "overall_status": "processing",
  "progress": 0.35,
  "completed_videos": 0,
  "failed_videos": 0,
  "total_api_cost": 12.50,
  "api_fallback_count": 1
}
```

### 4. Get Detailed Job Information

```bash
curl http://localhost:9000/batch/jobs/{job_id}/details
```

**Response:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "user_id": "user123",
  "name": "Multi-Language Dubbing",
  "overall_status": "processing",
  "progress": 0.35,
  "total_videos": 2,
  "completed_videos": 0,
  "failed_videos": 0,
  "total_api_cost": 12.50,
  "api_fallback_count": 1,
  "tasks": [
    {
      "task_id": "task-001",
      "video_path": "/videos/video1.mp4",
      "status": "translating",
      "progress": 0.45,
      "output_paths": {},
      "api_calls": [
        {
          "provider": "assemblyai",
          "task": "transcription",
          "status": "success",
          "timestamp": "2024-01-15T10:30:00",
          "cost": 0.5
        },
        {
          "provider": "deepl",
          "task": "translation_to_es",
          "status": "success",
          "timestamp": "2024-01-15T10:31:00",
          "cost": 0.25
        }
      ],
      "total_api_cost": 0.75,
      "error": null
    }
  ]
}
```

### 5. List Active Jobs

```bash
curl http://localhost:9000/batch/jobs
```

## Multi-API Fallback Logic

The agent automatically falls back to alternative providers in the following scenarios:

```
STT Task:
  1. Try AssemblyAI
  2. If fails → Try Deepgram
  3. If fails → Try OpenAI Whisper
  4. If fails → Use Local Whisper (no cost)

Translation Task:
  1. Try DeepL
  2. If fails → Try Google Translate
  3. If fails → Try Microsoft Translator
  4. If fails → Try OpenAI GPT
  5. If fails → Try Google Gemini
  6. If fails → Use Local Helsinki-NLP (no cost)

TTS Task:
  1. Try ElevenLabs
  2. If fails → Try Google Cloud TTS
  3. If fails → Try Azure TTS
  4. If fails → Use Local TTS (no cost)

Lip-Sync Task:
  1. Try Replicate
  2. If fails → Try Runway
  3. If fails → Use Local Wav2Lip (no cost)
```

## Cost Optimization

The agent tracks API costs and:

1. **Prioritizes Free Tiers**: Uses free tier APIs before paid ones
2. **Fallback to Local**: Always has local/free fallback options
3. **Cost Tracking**: Records cost for each API call
4. **Batch Reporting**: Shows total API cost per job

## Concurrent Processing

### Settings

```json
{
  "concurrent_tasks": 2,  // Process 2 videos simultaneously
  "timeout_per_video": 3600  // 1 hour timeout per video
}
```

### Performance Tips

1. Adjust `concurrent_tasks` based on:
   - Your machine's resources
   - API rate limits
   - Network bandwidth

2. Example configurations:
   ```
   Lightweight (Local only): 4 concurrent tasks
   Mixed (Some APIs): 2 concurrent tasks
   Heavy (All APIs): 1-2 concurrent tasks
   ```

## Monitoring and Logging

### Enable Debug Logging

In `ai-service/main.py`:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Log Files

All operations are logged to stdout. Redirect to file:
```bash
python -m uvicorn main:app > logs/batch_agent.log 2>&1 &
```

## Error Handling

The agent provides detailed error information:

```json
{
  "task_id": "task-001",
  "status": "failed",
  "error": "All TTS providers failed for language 'es'. Last error: API rate limit exceeded",
  "api_calls": [
    {"provider": "elevenlabs", "status": "failed", "error": "API rate limit exceeded"},
    {"provider": "google_tts", "status": "failed", "error": "Connection timeout"},
    {"provider": "azure_tts", "status": "failed", "error": "Invalid credentials"},
    {"provider": "local_tts", "status": "failed", "error": "Out of memory"}
  ]
}
```

## Production Deployment

### Docker Deployment

```dockerfile
FROM python:3.10-slim

RUN apt-get update && apt-get install -y ffmpeg
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .

EXPOSE 9000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "9000"]
```

### Docker Compose

```yaml
services:
  ai-service:
    build: ./ai-service
    ports:
      - "9000:9000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - DEEPL_API_KEY=${DEEPL_API_KEY}
      - ELEVENLABS_API_KEY=${ELEVENLABS_API_KEY}
    volumes:
      - ./uploads:/app/uploads
      - ./outputs:/app/outputs
```

## Troubleshooting

### Common Issues

1. **"All providers failed"**
   - Check API keys in .env
   - Verify API rate limits
   - Check internet connection

2. **"FFmpeg not found"**
   - Install FFmpeg for your OS
   - Add to system PATH

3. **Memory issues with local models**
   - Reduce concurrent_tasks
   - Use API providers instead of local models
   - Increase available RAM

4. **Slow processing**
   - Increase concurrent_tasks (if resources allow)
   - Use faster providers (check your free tier options)
   - Optimize video resolution/quality

## Support

For issues or questions:
1. Check the logs for detailed error messages
2. Review API provider documentation
3. Open an issue with job_id and error details
