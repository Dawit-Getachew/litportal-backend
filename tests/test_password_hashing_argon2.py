"""Argon2id password hashing with legacy bcrypt verification (core.security)."""
import asyncio
import time

import bcrypt
import pytest

from core.security import (
    hash_password,
    hash_password_async,
    verify_password,
    verify_password_async,
)

PASSWORD = "CorrectHorse9!Battery"


def _legacy_bcrypt(password: str, rounds: int = 4, ident: str = "$2b$") -> str:
    hashed = bcrypt.hashpw(password.encode("utf-8")[:72], bcrypt.gensalt(rounds)).decode()
    return hashed.replace("$2b$", ident, 1)


def test_new_hashes_are_argon2id_owasp_profile():
    hashed = hash_password(PASSWORD)
    assert hashed.startswith("$argon2id$v=19$m=19456,t=2,p=1$")
    assert verify_password(PASSWORD, hashed)
    assert not verify_password("WrongHorse9!Battery", hashed)


@pytest.mark.parametrize("ident", ["$2a$", "$2b$", "$2y$"])
def test_existing_bcrypt_hashes_still_verify(ident):
    legacy = _legacy_bcrypt(PASSWORD, ident=ident)
    assert verify_password(PASSWORD, legacy)
    assert not verify_password("WrongHorse9!Battery", legacy)


def test_existing_cost12_hash_from_previous_hash_password():
    # Exactly what the previous hash_password wrote: bcrypt.gensalt() == cost 12.
    legacy = bcrypt.hashpw(PASSWORD.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    assert legacy.startswith("$2b$12$")
    assert verify_password(PASSWORD, legacy)


@pytest.mark.parametrize("bad_hash", ["", "garbage", "$argon2id$nope", "$2b$12$short", None])
def test_bad_hashes_return_false(bad_hash):
    # Shadow users store "" and must keep getting a 401, never a 500.
    assert verify_password(PASSWORD, bad_hash) is False


def test_async_variants_roundtrip_and_do_not_block():
    legacy = _legacy_bcrypt(PASSWORD, rounds=12)

    async def run():
        hashed = await hash_password_async(PASSWORD)
        assert await verify_password_async(PASSWORD, hashed)
        ticks = 0
        stop = asyncio.Event()

        async def ticker():
            nonlocal ticks
            while not stop.is_set():
                await asyncio.sleep(0.005)
                ticks += 1

        task = asyncio.create_task(ticker())
        started = time.perf_counter()
        assert await verify_password_async(PASSWORD, legacy)
        elapsed = time.perf_counter() - started
        stop.set()
        await task
        return ticks, elapsed

    ticks, elapsed = asyncio.run(run())
    assert elapsed > 0.02
    assert ticks >= 3
