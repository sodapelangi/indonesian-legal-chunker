# Indonesian Legal Document Chunker API

A FastAPI service for chunking Indonesian legal documents into hierarchical levels with progress tracking.

## Features

- **3-Level Hierarchical Chunking**:
  - Level 1: Document metadata (Judul, Nomor, Tahun, etc.)
  - Level 2: Major sections (BAB, LAMPIRAN, BAGIAN)
  - Level 3: Articles (Pasal)

- **Real-time Progress Tracking**: Monitor processing progress with detailed updates
- **Comprehensive Metadata Extraction**: Automatically extracts Indonesian regulation fields
- **Flexible Chunking Options**: Configurable chunk size and overlap
- **Robust Validation**: Input validation and error handling
- **RESTful API**: Clean API design with proper HTTP status codes

## Quick Start

### Using Docker (Recommended)

1. **Build and run with Docker Compose**:
```bash
docker-compose up --build
```

2. **Test the API**:
```bash
python test_api.py
```

### Manual Setup

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Run the server**:
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

3. **Access the API**:
- API Documentation: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

## API Endpoints

### POST /chunk-document
Start document processing with progress tracking.

**Request**:
```json
{
  "text": "PERATURAN MENTERI...",
  "options": {
    "max_chunk_size": 1500,
    "overlap_size": 100
  }
}
```

**Response**:
```json
{
  "status": "processing",
  "job_id": "uuid-here"
}
```

### GET /job/{job_id}/progress
Get real-time processing progress.

**Response**:
```json
{
  "job_id": "uuid-here",
  "stage": "chunking",
  "progress": 65.0,
  "current_chunk": 12,
  "total_chunks": 18,
  "estimated_time_remaining": 8.5,
  "message": "Processing section: BAB II"
}
```

### GET /job/{job_id}/result
Get final processing result.

**Response**:
```json
{
  "status": "completed",
  "job_id": "uuid-here",
  "processing_time": 12.5,
  "metadata": {
    "judul": "PERATURAN MENTERI...",
    "nomor": "14 TAHUN 2024",
    "tahun": "2024",
    "tentang": "PENYELENGGARAAN PENGAWASAN...",
    "menimbang": [...],
    "mengingat": [...]
  },
  "chunks": [
    {
      "level": 1,
      "title": "Metadata Dokumen",
      "content": "Judul: PERATURAN MENTERI...",
      "chunk_type": "metadata",
      "section_number": null,
      "parent_section": null,
      "content_length": 1234
    }
  ],
  "summary": {
    "total_chunks": 25,
    "level_counts": {"1": 1, "2": 5, "3": 19},
    "type_counts": {"metadata": 1, "major_section": 5, "pasal": 19},
    "total_characters": 45000
  }
}
```

## Processing Stages

1. **validation** (5%): Document format validation
2. **parsing** (15-25%): Metadata extraction and structure analysis
3. **chunking** (40-90%): Creating hierarchical chunks with progress per section
4. **finalizing** (95%): Preparing response
5. **completed** (100%): Processing finished

## Document Structure Support

The chunker recognizes Indonesian legal document patterns:

- **Metadata**: PERATURAN, NOMOR, TAHUN, TENTANG, Menimbang, Mengingat
- **Major Sections**: BAB I, LAMPIRAN I, BAGIAN KESATU
- **Articles**: Pasal 1, Pasal 2a, etc.
- **Hierarchical Relationships**: Automatic parent-child linking

## Configuration

### Environment Variables
- `PYTHONPATH`: Set to `/app`
- `PYTHONUNBUFFERED`: Set to `1` for real-time logging

### Chunking Options
- `max_chunk_size`: Maximum characters per chunk (100-5000)
- `overlap_size`: Character overlap between chunks (0-500)

## Error Handling

The API includes comprehensive validation:
- Document format validation
- Indonesian legal document pattern detection
- Character encoding validation
- Size limits and timeout handling

## Production Deployment

### Docker Deployment
```bash
# Build production image
docker build -t chunker-api .

# Run with resource limits
docker run -d \
  --name chunker-api \
  -p 8000:8000 \
  --memory=2g \
  --cpus=1 \
  chunker-api
```

### Cloud Deployment Options
- **Google Cloud Run**: Serverless container deployment
- **Railway**: Simple git-based deployment
- **Render**: Container deployment with auto-scaling
- **AWS ECS**: Enterprise container orchestration

## Monitoring

- Health check endpoint: `/health`
- Structured logging with job tracking
- Processing time metrics
- Memory usage monitoring via Docker stats

## Integration with Supabase

This service is designed to integrate with Supabase Edge Functions:

1. Edge Function receives file upload
2. Calls this chunker API
3. Processes response and stores in database
4. Updates frontend with progress via WebSocket

## Testing

Run the comprehensive test suite:
```bash
python test_api.py
```

The test includes:
- Health check validation
- Document processing workflow
- Progress monitoring
- Result verification
- Sample output display

## Support

For issues or questions:
1. Check the API documentation at `/docs`
2. Review logs for error details
3. Test with the provided sample document
4. Verify all dependencies are installed correctly