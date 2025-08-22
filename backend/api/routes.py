from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from backend.api.models import ResearchRequest, SearchRequest
from backend.api import db, crew

router = APIRouter()

@router.post("/start-job-stream")
async def start_research_job_stream(request: ResearchRequest):
    async def event_generator():
        # Researcher
        yield f"data: {crew.stream_event('stage', 'researcher:start')}\n\n"
        researcher_notes = await crew.run_researcher_async(request.topic)
        yield f"data: {crew.stream_event('stage', 'researcher:done')}\n\n"

        # Analyst
        yield f"data: {crew.stream_event('stage', 'analyst:start')}\n\n"
        analyst_notes = await crew.run_analyst_async(researcher_notes)
        yield f"data: {crew.stream_event('stage', 'analyst:done')}\n\n"

        # Writer: token stream
        yield f"data: {crew.stream_event('stage', 'writer:start')}\n\n"
        final_accum = []
        async for token in crew.run_writer_token_stream(
            topic=request.topic,
            researcher_notes=researcher_notes,
            analyst_notes=analyst_notes,
        ):
            final_accum.append(token)
            yield f"data: {crew.stream_event('token', token)}\n\n"

        full_text = "".join(final_accum)
        yield f"data: {crew.stream_event('stage', 'writer:done')}\n\n"

        # Save exactly what was streamed
        report_id = crew.generate_report_id()
        db.save_report(report_id, full_text, title=request.topic)

        # Final event (build dict first to avoid f-string brace issues)
        meta = {"report_id": report_id, "title": request.topic}
        yield f"data: {crew.stream_event('final', meta)}\n\n"
        yield "event: close\ndata: done\n\n"

    # Important for SSE: ensure streaming MIME and no buffering on your proxy
    return StreamingResponse(event_generator(), media_type="text/event-stream")

@router.get("/list-reports")
async def list_reports():
    return db.list_reports()

@router.post("/search-reports")
async def search_reports(request: SearchRequest):
    return db.search_reports(request.query)
