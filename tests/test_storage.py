"""Tests for storage backends."""

import pytest

from cbreaker.storage.memory import MemoryStorage


class TestMemoryStorage:
    """Tests for MemoryStorage."""

    def test_set_and_get(self) -> None:
        """Should store and retrieve data."""
        storage = MemoryStorage()
        state = {"key": "value", "count": 42}

        storage.set("test_key", state)
        result = storage.get("test_key")

        assert result == state

    def test_get_nonexistent_key(self) -> None:
        """Should return None for nonexistent key."""
        storage = MemoryStorage()
        result = storage.get("nonexistent")
        assert result is None

    def test_delete(self) -> None:
        """Should delete stored data."""
        storage = MemoryStorage()
        storage.set("test_key", {"data": "value"})

        storage.delete("test_key")
        result = storage.get("test_key")

        assert result is None

    def test_delete_nonexistent_key(self) -> None:
        """Delete should not raise for nonexistent key."""
        storage = MemoryStorage()
        storage.delete("nonexistent")  # Should not raise

    def test_clear(self) -> None:
        """Should clear all data."""
        storage = MemoryStorage()
        storage.set("key1", {"a": 1})
        storage.set("key2", {"b": 2})

        storage.clear()

        assert storage.get("key1") is None
        assert storage.get("key2") is None
        assert storage.keys() == []

    def test_keys(self) -> None:
        """Should return all stored keys."""
        storage = MemoryStorage()
        storage.set("key1", {"a": 1})
        storage.set("key2", {"b": 2})

        keys = storage.keys()

        assert set(keys) == {"key1", "key2"}

    def test_state_isolation(self) -> None:
        """Stored state should be isolated from modifications."""
        storage = MemoryStorage()
        original = {"count": 1}
        storage.set("key", original)

        # Modify original
        original["count"] = 999

        # Retrieved should still have original value
        result = storage.get("key")
        assert result["count"] == 1

    @pytest.mark.asyncio
    async def test_async_operations(self) -> None:
        """Async operations should work."""
        storage = MemoryStorage()
        state = {"async": True}

        await storage.set_async("async_key", state)
        result = await storage.get_async("async_key")

        assert result == state

        await storage.delete_async("async_key")
        result = await storage.get_async("async_key")

        assert result is None
