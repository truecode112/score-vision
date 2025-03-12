import asyncio

class AsyncBarrier:
    """
    An asyncio-based barrier to synchronize multiple asynchronous tasks.
    Ensures that all parties reach the barrier before proceeding.
    """

    def __init__(self, parties: int):
        """
        Initialize the barrier.
        
        Args:
            parties (int): The number of tasks that must reach the barrier before proceeding.
        """
        self.parties = parties
        self.count = 0
        self.condition = asyncio.Condition()
        self.generation = 0  # To allow reuse of the barrier

    async def wait(self):
        """
        Wait until all parties have reached the barrier.
        Once all parties reach the barrier, they are released simultaneously.
        """
        async with self.condition:
            gen = self.generation
            self.count += 1
            if self.count == self.parties:
                # All parties have reached the barrier.
                self.generation += 1
                self.count = 0
                self.condition.notify_all()
            else:
                # Wait until the barrier is released.
                while gen == self.generation:
                    await self.condition.wait()
