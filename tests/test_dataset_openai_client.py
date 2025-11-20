import asyncio
import time

import pytest

from inference_perf.client.modelserver.dataset_openai_client import _DetailedResultPublisher


class _SlowQueue:
    def __init__(self) -> None:
        self.items: list[tuple] = []

    def put(self, payload: tuple) -> None:
        time.sleep(0.05)
        self.items.append(payload)


@pytest.mark.asyncio
async def test_detailed_result_publisher_does_not_block_event_loop() -> None:
    queue = _SlowQueue()
    publisher = _DetailedResultPublisher(queue, max_pending=10)

    start = time.perf_counter()
    for i in range(5):
        publisher.publish((f"req-{i}", 200, 0.0, 0.0, 0.0, [], 0))
    elapsed = time.perf_counter() - start

    # Publishing should be effectively instantaneous even though the
    # underlying queue blocks for a noticeable amount of time.
    assert elapsed < 0.01

    await asyncio.sleep(0.4)
    assert len(queue.items) == 5

    await publisher.close()
