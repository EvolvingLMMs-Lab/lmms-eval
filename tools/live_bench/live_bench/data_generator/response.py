class Response(object):
    def __init__(self, success: bool, content: str, full_log: dict):
        self.success = success
        self.content = content
        self.full_log = full_log

    def to_dict(self):
        return {
            "success": self.success,
            "content": self.content,
            "full_log": self.full_log,
        }
