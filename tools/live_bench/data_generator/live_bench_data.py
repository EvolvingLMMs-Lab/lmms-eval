from lmms_eval.live_bench.screen_shoter.screen import ScreenImage
import datasets


class LiveBenchData(object):
    features = datasets.Features(
        {
            "id": datasets.Value("int32"),
            "images": datasets.Sequence(datasets.Image()),
            "website": datasets.Value("string"),
            "question": datasets.Value("string"),
            "answer": datasets.Value("string"),
            "subtask": datasets.Value("string"),
            "data_generator": datasets.Value("string"),
            "checker": datasets.Value("string"),
            "date_time": datasets.Value("string"),
            "screen_shoter": datasets.Value("string"),
            "screen_size": datasets.Value("string"),
            "score": datasets.Value("int32"),
            "reason": datasets.Value("string"),
            "scorer_name": datasets.Value("string"),
        }
    )

    def __init__(self, *, screen: ScreenImage, question: str, answer: str, subtask: str, data_generator: str, score: int = None, reason: str = None, checker=None, scorer_name=None, scorer=None):
        self.screen = screen
        self.question = question
        self.answer = answer
        self.subtask = subtask
        self.data_generator = data_generator
        self.checker = None
        if checker is not None:
            response = checker.check(screen, question, answer, subtask)
            if response.success:
                formatted_response = checker.format_checked_response(response)
                if formatted_response.question and formatted_response.answer:
                    self.question = formatted_response.question
                    self.answer = formatted_response.answer
                    if formatted_response.subtask:
                        self.subtask = formatted_response.subtask
                    else:
                        self.subtask = subtask
                    self.checker = checker.get_name()
        if score is not None:
            self.score = score
            self.reason = reason
            self.scorer_name = scorer_name
        else:
            score = scorer.get_score(question, answer, screen.images)
            self.score = score.score
            self.reason = score.reason
            self.scorer_name = scorer.get_name()

    def to_dict(self):
        images = self.screen.images
        website = self.screen.website.get_info()
        question = self.question
        answer = self.answer
        subtask = self.subtask
        data_generator = self.data_generator
        date_time = self.screen.capture_datetime
        screen_shoter = self.screen.shoter
        screen_size = self.screen.screen_size
        return {
            "images": images,
            "website": website,
            "question": question,
            "answer": answer,
            "subtask": subtask,
            "data_generator": data_generator,
            "checker": self.checker,
            "date_time": date_time,
            "screen_shoter": screen_shoter,
            "screen_size": screen_size,
            "score": self.score,
            "reason": self.reason,
            "scorer_name": self.scorer_name,
        }

    def to_hf_dict(self):
        return self.features.encode_example(self.to_dict())

    def to_output_dict(self):
        return {
            "screen": self.screen.to_output_dict(),
            "question": self.question,
            "answer": self.answer,
            "subtask": self.subtask,
            "data_generator": self.data_generator,
            "checker": self.checker,
            "score": self.score,
            "reason": self.reason,
            "scorer_name": self.scorer_name,
        }
