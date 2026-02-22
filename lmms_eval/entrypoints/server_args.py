from dataclasses import dataclass, field


@dataclass
class ServerArgs:
    """Arguments for launching the evaluation server."""

    host: str = field(default="localhost")
    port: int = field(default=8000)
    max_completed_jobs: int = field(default=100)
    temp_dir_prefix: str = field(default="lmms_eval_")

    def __post_init__(self) -> None:
        """Validate ServerArgs fields after initialization."""
        if not isinstance(self.port, int) or not (1 <= self.port <= 65535):
            raise ValueError(f"Port must be an integer between 1 and 65535, " f"got {self.port!r} (type: {type(self.port).__name__})")
        if not isinstance(self.max_completed_jobs, int) or self.max_completed_jobs < 1:
            raise ValueError(f"max_completed_jobs must be a positive integer, " f"got {self.max_completed_jobs!r}")

    @classmethod
    def from_dict(cls, d: dict) -> "ServerArgs":
        """Create ServerArgs from a dictionary."""
        return cls(
            host=d.get("host", "localhost"),
            port=d.get("port", 8000),
            max_completed_jobs=d.get("max_completed_jobs", 100),
            temp_dir_prefix=d.get("temp_dir_prefix", "lmms_eval_"),
        )

    def to_dict(self) -> dict:
        """Convert ServerArgs to a dictionary."""
        return {
            "host": self.host,
            "port": self.port,
            "max_completed_jobs": self.max_completed_jobs,
            "temp_dir_prefix": self.temp_dir_prefix,
        }
