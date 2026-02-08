class FixedTimeAgent:
    """Baseline agent that cycles through actions in a fixed order."""

    def __init__(self, num_actions: int = 4) -> None:
        """Initialize the baseline agent.

        Args:
            num_actions: Number of discrete actions to cycle through.
        """
        self.num_actions = num_actions
        self._step = 0

    def choose_action(self, state: object) -> int:  # noqa: ARG002
        """Return the next action in the fixed cycle."""
        action = self._step % self.num_actions
        self._step += 1
        return action
