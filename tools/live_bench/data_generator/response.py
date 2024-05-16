class Response(object):
    def __init__(self, success: bool, content: str, full_log: dict):
        self.success = success
        self.content = content
        self.full_log = full_log
