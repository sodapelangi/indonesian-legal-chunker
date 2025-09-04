from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import asyncio
import time
import uuid
import json
from datetime import datetime
import logging

# Import your chunker
from chunker import IndonesianLegalDocumentChunker, DocumentChunk

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Indonesian Legal Document Chunker API",
    description="API service for chunking Indonesian legal documents",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class ChunkingOptions(BaseModel):
    max_chunk_size: int = Field(default=1500, ge=100, le=5000)
    overlap_size: int = Field(default=100, ge=0, le=500)

class ChunkDocumentRequest(BaseModel):
    text: str = Field(..., min_length=10, max_length=10_000_000)  # 10MB text limit
    options: Optional[ChunkingOptions] = ChunkingOptions()
    job_id: Optional[str] = None

class ChunkResponse(BaseModel):
    level: int
    title: str
    content: str
    chunk_type: Optional[str]
    section_number: Optional[str]
    parent_section: Optional[str]
    content_length: int

class ProcessingResponse(BaseModel):
    status: str
    job_id: str
    processing_time: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    chunks: Optional[List[ChunkResponse]] = None
    summary: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class ProgressUpdate(BaseModel):
    job_id: str
    stage: str
    progress: float
    current_chunk: Optional[int] = None
    total_chunks: Optional[int] = None
    estimated_time_remaining: Optional[float] = None
    message: str

# In-memory storage for job progress (use Redis in production)
job_progress: Dict[str, ProgressUpdate] = {}
job_results: Dict[str, ProcessingResponse] = {}

def validate_document_text(text: str) -> None:
    """Basic validation for Indonesian legal documents"""
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="Document text cannot be empty")
    
    # Check for minimum Indonesian legal document indicators
    text_upper = text.upper()
    legal_indicators = [
        'PERATURAN', 'UNDANG-UNDANG', 'KEPUTUSAN', 'INSTRUKSI', 
        'NOMOR', 'TAHUN', 'TENTANG', 'PASAL'
    ]
    
    found_indicators = sum(1 for indicator in legal_indicators if indicator in text_upper)
    if found_indicators < 3:
        raise HTTPException(
            status_code=400, 
            detail="Text does not appear to be an Indonesian legal document"
        )
    
    # Check encoding issues
    try:
        text.encode('utf-8')
    except UnicodeEncodeError:
        raise HTTPException(status_code=400, detail="Document contains invalid characters")

def estimate_processing_time(text_length: int) -> float:
    """Estimate processing time based on document length"""
    # Rough estimate: 1000 characters per second
    base_time = text_length / 1000
    # Add overhead for parsing and chunking
    return max(5.0, base_time * 1.5)

async def update_progress(job_id: str, stage: str, progress: float, message: str, 
                         current_chunk: Optional[int] = None, total_chunks: Optional[int] = None):
    """Update job progress"""
    estimated_time = None
    if job_id in job_progress:
        start_time = job_progress[job_id].estimated_time_remaining or time.time()
        if progress > 0:
            elapsed = time.time() - start_time
            estimated_time = (elapsed / progress) * (100 - progress)
    
    job_progress[job_id] = ProgressUpdate(
        job_id=job_id,
        stage=stage,
        progress=progress,
        current_chunk=current_chunk,
        total_chunks=total_chunks,
        estimated_time_remaining=estimated_time,
        message=message
    )
    logger.info(f"Job {job_id}: {stage} - {progress:.1f}% - {message}")

async def process_document_async(job_id: str, text: str, options: ChunkingOptions):
    """Process document asynchronously with progress updates"""
    start_time = time.time()
    
    try:
        await update_progress(job_id, "validation", 5.0, "Validating document format")
        
        # Initialize chunker
        chunker = IndonesianLegalDocumentChunker(
            max_chunk_size=options.max_chunk_size,
            overlap_size=options.overlap_size
        )
        
        await update_progress(job_id, "parsing", 15.0, "Extracting document metadata")
        
        # Extract metadata
        metadata = chunker.extract_metadata(text)
        
        await update_progress(job_id, "parsing", 25.0, "Analyzing document structure")
        
        # Split into sections
        sections = chunker.split_into_sections(text)
        sections = chunker.establish_hierarchy(sections)
        
        await update_progress(job_id, "chunking", 35.0, "Creating metadata chunk")
        
        # Create chunks
        all_chunks = [chunker.create_metadata_chunk()]
        total_sections = len(sections)
        
        await update_progress(job_id, "chunking", 40.0, f"Processing {total_sections} sections", 0, total_sections)
        
        # Process each section with progress updates
        for i, section in enumerate(sections):
            if section['content']:
                section_chunks = chunker.chunk_long_content(section['content'], section)
                all_chunks.extend(section_chunks)
                
                progress = 40.0 + (50.0 * (i + 1) / total_sections)
                await update_progress(
                    job_id, "chunking", progress, 
                    f"Processed section: {section.get('title', 'Unknown')}", 
                    i + 1, total_sections
                )
                
                # Small delay to allow progress updates
                await asyncio.sleep(0.1)
        
        await update_progress(job_id, "finalizing", 95.0, "Preparing response")
        
        # Convert to response format
        chunk_responses = []
        for chunk in all_chunks:
            chunk_responses.append(ChunkResponse(
                level=chunk.level,
                title=chunk.title,
                content=chunk.content,
                chunk_type=chunk.chunk_type,
                section_number=chunk.section_number,
                parent_section=chunk.parent_section,
                content_length=len(chunk.content)
            ))
        
        # Create summary
        level_counts = {}
        type_counts = {}
        total_chars = 0
        
        for chunk in all_chunks:
            level_counts[str(chunk.level)] = level_counts.get(str(chunk.level), 0) + 1
            chunk_type = chunk.chunk_type or 'unknown'
            type_counts[chunk_type] = type_counts.get(chunk_type, 0) + 1
            total_chars += len(chunk.content)
        
        summary = {
            "total_chunks": len(all_chunks),
            "level_counts": level_counts,
            "type_counts": type_counts,
            "total_characters": total_chars
        }
        
        processing_time = time.time() - start_time
        
        # Store result
        job_results[job_id] = ProcessingResponse(
            status="completed",
            job_id=job_id,
            processing_time=processing_time,
            metadata=metadata,
            chunks=chunk_responses,
            summary=summary
        )
        
        await update_progress(job_id, "completed", 100.0, f"Processing completed in {processing_time:.1f}s")
        
    except Exception as e:
        logger.error(f"Error processing job {job_id}: {str(e)}")
        job_results[job_id] = ProcessingResponse(
            status="failed",
            job_id=job_id,
            error=str(e)
        )
        await update_progress(job_id, "failed", 0.0, f"Processing failed: {str(e)}")

@app.post("/chunk-document", response_model=ProcessingResponse)
async def chunk_document(request: ChunkDocumentRequest, background_tasks: BackgroundTasks):
    """Start document chunking process"""
    
    # Validate input
    validate_document_text(request.text)
    
    # Generate job ID
    job_id = request.job_id or str(uuid.uuid4())
    
    # Estimate processing time
    estimated_time = estimate_processing_time(len(request.text))
    
    # Initialize progress
    job_progress[job_id] = ProgressUpdate(
        job_id=job_id,
        stage="queued",
        progress=0.0,
        estimated_time_remaining=estimated_time,
        message="Document queued for processing"
    )
    
    # Start background processing
    background_tasks.add_task(process_document_async, job_id, request.text, request.options)
    
    return ProcessingResponse(
        status="processing",
        job_id=job_id
    )

@app.get("/job/{job_id}/progress", response_model=ProgressUpdate)
async def get_job_progress(job_id: str):
    """Get processing progress for a job"""
    if job_id not in job_progress:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return job_progress[job_id]

@app.get("/job/{job_id}/result", response_model=ProcessingResponse)
async def get_job_result(job_id: str):
    """Get processing result for a job"""
    if job_id not in job_results:
        # Check if job is still in progress
        if job_id in job_progress:
            return ProcessingResponse(
                status="processing",
                job_id=job_id
            )
        raise HTTPException(status_code=404, detail="Job not found")
    
    return job_results[job_id]

@app.delete("/job/{job_id}")
async def delete_job(job_id: str):
    """Clean up job data"""
    job_progress.pop(job_id, None)
    job_results.pop(job_id, None)
    return {"message": "Job data cleaned up"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_jobs": len(job_progress),
        "completed_jobs": len(job_results)
    }

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "service": "Indonesian Legal Document Chunker API",
        "version": "1.0.0",
        "endpoints": {
            "chunk_document": "POST /chunk-document",
            "job_progress": "GET /job/{job_id}/progress",
            "job_result": "GET /job/{job_id}/result",
            "health": "GET /health"
        }
    }

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
