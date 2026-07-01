"""User profile router."""
from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends

from core.db import Collections, get_db
from core.security import get_current_user
from models.users import UserProfileUpdate, UserPublic

router = APIRouter(prefix="/users", tags=["users"])


def _public(u: dict) -> UserPublic:
    return UserPublic(
        id=u["id"],
        email=u["email"],
        name=u["name"],
        role=u["role"],
        first_use_acknowledged=u.get("first_use_acknowledged", False),
        created_at=u["created_at"],
        updated_at=u["updated_at"],
    )


@router.get("/me/profile", response_model=UserPublic)
async def get_profile(current_user: dict = Depends(get_current_user)):
    return _public(current_user)


@router.patch("/me/profile", response_model=UserPublic)
async def patch_profile(
    body: UserProfileUpdate,
    current_user: dict = Depends(get_current_user),
):
    db = get_db()
    update: dict = {}
    if body.name is not None:
        update["name"] = body.name.strip()
    if update:
        update["updated_at"] = datetime.now(timezone.utc)
        await db[Collections.USERS].update_one({"id": current_user["id"]}, {"$set": update})
    fresh = await db[Collections.USERS].find_one({"id": current_user["id"]}, {"_id": 0})
    return _public(fresh)
