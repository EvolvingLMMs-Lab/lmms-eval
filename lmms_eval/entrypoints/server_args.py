from dataclasses import dataclass, field


@dataclass
class ServerArgs:
    """Arguments for launching the evaluation server."""

    host: str = field(default="localhost")
    port: int = field(default=8000)

    def __post_init__(self):
        if not isinstance(self.port, int) or not (1 <= self.port <= 65535):
            raise ValueError(
                f"Port must be an integer between 1 and 65535, got {self.port}"
            )

    @classmethod
    def from_dict(cls, d: dict) -> "ServerArgs":
        """Create ServerArgs from a dictionary."""
        return cls(
            host=d.get("host", "localhost"),
            port=d.get("port", 8000),
        )

    def to_dict(self) -> dict:
        """Convert ServerArgs to a dictionary."""
        return {
            "host": self.host,
            "port": self.port,
        }
