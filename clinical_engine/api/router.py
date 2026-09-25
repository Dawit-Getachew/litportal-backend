"""Versioned engine API: /api/v2/engine/* (also /api/litportal/v2/engine/*).

Additive only — no existing route or shape changes. Every endpoint except
/capabilities is authenticated, owner-scoped, and answers 404 while
CLINICAL_ENGINE_ENABLED is off.
"""
from __future__ import annotations

from typing import Any, Awaitable, Literal, Optional, TypeVar

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, ConfigDict, Field

from clinical_engine import ENGINE_VERSION, questions
from clinical_engine.config import settings
from clinical_engine.contracts.question import AGE_GROUPS
from clinical_engine.gateway import ModelGateway
from clinical_engine.interpretation.present import AGE_LABELS, TYPE_LABELS, ChipEdit, EngineInputError
from clinical_engine.questions import DECLINE_REASONS, EngineConflict, EngineNotFound
from core.security import get_current_user

router = APIRouter(prefix="/v2/engine", tags=["clinical-engine"])
T = TypeVar("T")


def _enabled() -> None:
    if not settings.ENABLED:
        raise HTTPException(status_code=404, detail="Not Found")


async def _call(awaitable: Awaitable[T]) -> T:
    try:
        return await awaitable
    except EngineNotFound:
        raise HTTPException(status_code=404, detail={"code": "not_found", "message": "Not found."})
    except EngineConflict as exc:
        raise HTTPException(status_code=409, detail={"code": exc.code, "message": exc.message})
    except EngineInputError as exc:
        raise HTTPException(status_code=422, detail={"code": exc.code, "message": exc.message})


class CreateQuestionBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1, max_length=5000)


class ReviseQuestionBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    base_version: int = Field(ge=1)
    text: Optional[str] = Field(default=None, max_length=5000)
    edits: Optional[list[ChipEdit]] = Field(default=None, max_length=30)


class ConfirmQuestionBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: int = Field(ge=1)
    mode: Literal["review_papers", "answer_for_me"]


class DeclineQuestionBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reason: Literal["wrong_interpretation", "new_question", "other"] = "new_question"


@router.get("/capabilities")
async def capabilities() -> dict[str, Any]:
    return {
        "enabled": bool(settings.ENABLED),
        "engine_version": ENGINE_VERSION,
        "modes": ["review_papers", "answer_for_me"],
        "retrieval_adapter": settings.RETRIEVAL_ADAPTER if settings.ENABLED else None,
        "model_configured": ModelGateway().is_configured() if settings.ENABLED else False,
        "question_types": [{"value": k, "label": v} for k, v in TYPE_LABELS.items()],
        "age_groups": [{"value": a, "label": AGE_LABELS[a]} for a in AGE_GROUPS],
        "decline_reasons": list(DECLINE_REASONS),
        "limits": {"max_question_chars": settings.MAX_QUESTION_CHARS,
                   "max_versions_per_question": settings.MAX_VERSIONS_PER_QUESTION},
    }


@router.post("/questions", status_code=status.HTTP_201_CREATED, dependencies=[Depends(_enabled)])
async def create_question(body: CreateQuestionBody, current_user: dict = Depends(get_current_user)):
    return await _call(questions.create_question(current_user, body.text))


@router.get("/questions", dependencies=[Depends(_enabled)])
async def list_questions(limit: int = Query(20, ge=1, le=50), current_user: dict = Depends(get_current_user)):
    return {"questions": await _call(questions.list_questions(current_user, limit))}


@router.get("/questions/{question_id}", dependencies=[Depends(_enabled)])
async def get_question(question_id: str, current_user: dict = Depends(get_current_user)):
    return await _call(questions.get_question(current_user, question_id))


@router.post("/questions/{question_id}/revise", dependencies=[Depends(_enabled)])
async def revise_question(question_id: str, body: ReviseQuestionBody,
                          current_user: dict = Depends(get_current_user)):
    return await _call(questions.revise_question(
        current_user, question_id, base_version=body.base_version, text=body.text, edits=body.edits,
    ))


@router.post("/questions/{question_id}/confirm", dependencies=[Depends(_enabled)])
async def confirm_question(question_id: str, body: ConfirmQuestionBody,
                           current_user: dict = Depends(get_current_user)):
    return await _call(questions.confirm_question(current_user, question_id, version=body.version, mode=body.mode))


@router.post("/questions/{question_id}/decline", dependencies=[Depends(_enabled)])
async def decline_question(question_id: str, body: DeclineQuestionBody,
                           current_user: dict = Depends(get_current_user)):
    return await _call(questions.decline_question(current_user, question_id, body.reason))


@router.get("/questions/{question_id}/trace", dependencies=[Depends(_enabled)])
async def question_trace(question_id: str, current_user: dict = Depends(get_current_user)):
    return await _call(questions.question_trace(current_user, question_id))


@router.get("/runs/{run_id}", dependencies=[Depends(_enabled)])
async def get_run(run_id: str, current_user: dict = Depends(get_current_user)):
    return await _call(questions.get_run(current_user, run_id))


@router.post("/runs/{run_id}/{action}", dependencies=[Depends(_enabled)])
async def control_run(run_id: str, action: Literal["pause", "resume", "cancel"],
                      current_user: dict = Depends(get_current_user)):
    return await _call(questions.control_run(current_user, run_id, action))


@router.get("/runs/{run_id}/trace", dependencies=[Depends(_enabled)])
async def run_trace(run_id: str, current_user: dict = Depends(get_current_user)):
    return await _call(questions.run_trace(current_user, run_id))


@router.get("/evidence-sets/{set_id}", dependencies=[Depends(_enabled)])
async def get_evidence_set(set_id: str, current_user: dict = Depends(get_current_user)):
    return await _call(questions.get_evidence_set(current_user, set_id))
