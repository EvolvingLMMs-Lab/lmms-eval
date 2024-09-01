import json
import logging
import os
from datetime import datetime
from typing import List, Tuple

from datasets import Dataset, load_dataset
from live_bench.data_generator import get_generator, get_random_generator
from live_bench.data_generator.live_bench_data import LiveBenchData
from live_bench.data_generator.qa_generator import QAData, QAGenerator
from live_bench.data_generator.question_finalizer import QuestionFinalizer
from live_bench.data_generator.response import Response
from live_bench.data_generator.score_getter import (
    get_random_score_getter,
    get_score_getter,
)
from live_bench.data_generator.utils.extract_infomation import (
    ImageInfomation,
    InfomationExtractor,
)
from live_bench.driver import load_driver
from live_bench.screen_shoter import ScreenImage, ScreenShoter, get_shoter
from live_bench.websites import Website
from tqdm import tqdm

logger = logging.getLogger("lmms-eval")


def get_qa_data(images: ScreenImage, qa_generator: QAGenerator, *, infomation_getter: InfomationExtractor = None, test=False) -> Tuple[List[QAData], Response]:
    if infomation_getter:
        infomation = infomation_getter.extract_infomation(images)
    else:
        infomation = None
    response = qa_generator.generate(images, test=test, infomation=infomation)
    qa_data = qa_generator.format_response(response)
    return qa_data, response


def get_live_bench_data(
    driver, website: Website, screen_shoter: ScreenShoter, qa_generator: QAGenerator, checker: QAGenerator, infomation_getter: InfomationExtractor, question_finalizer: QuestionFinalizer, test=False, scorer=None, score_threshold=5
) -> Tuple[List[LiveBenchData], Response]:
    images = screen_shoter.capture(driver, website)
    qa_data, logs = get_qa_data(images, qa_generator, test=test, infomation_getter=infomation_getter)
    data = []
    for qa in qa_data:
        # qa_data = question_finalizer.finalize_question(qa, images.images)
        item = LiveBenchData(screen=images, question=qa.question, answer=qa.answer, subtask=qa.subtask, criteria=qa.criteria, data_generator=qa_generator.get_name(), checker=checker, scorer=scorer, finalizer=question_finalizer)
        if score_threshold and (not item.score or item.score < score_threshold):
            continue
        data.append(item)
    return data, logs


class LiveBench(object):
    def __init__(self, path: str = "lmms-lab/LiveBench", *, name="auto", split="test", cache_dir=None, remote_path=None, trust_remote_code=True, force_clear=False, **kwargs):
        self.path = path
        if name == "auto":
            name = datetime.now().strftime("%Y-%m")
        self.name = name
        self.split = split
        self.cache_dir = cache_dir
        self.dataset_kwargs = kwargs
        if remote_path is None:
            self.remote_path = path
        if force_clear:
            self.clear()
        else:
            try:
                self.hf_data = load_dataset(self.path, name=self.name, split=split, cache_dir=cache_dir, trust_remote_code=trust_remote_code, **kwargs)
            except Exception as e:
                logger.error(f"Error loading dataset: {e}")
                self.clear()

    def clear(self):
        self.hf_data = Dataset.from_dict(
            {
                "id": [],
                "images": [],
                "website": [],
                "question": [],
                "answer": [],
                "criteria": [],
                "subtask": [],
                "data_generator": [],
                "checker": [],
                "date_time": [],
                "screen_shoter": [],
                "screen_size": [],
                "score": [],
                "reason": [],
                "scorer_name": [],
            },
            features=LiveBenchData.features,
        )

    def add(self, data: LiveBenchData, id: int = None):
        if id is None:
            id = len(self.hf_data)
        organized_data = data.to_hf_dict()
        organized_data["id"] = id
        self.hf_data = self.hf_data.add_item(organized_data)

    def capture(
        self,
        websites: List[Website] = None,
        *,
        screen_shoter="single_screen",
        qa_generator=None,
        checker=None,
        driver=None,
        scorer=None,
        question_finalizer=None,
        test=False,
        driver_kwargs={},
        shoter_kwargs={},
        generator_kwargs={},
        question_finalizer_kwargs={},
        log_folder="./logs",
    ):
        can_quit_driver = False
        if driver is None and screen_shoter != "human":
            driver = load_driver(**driver_kwargs)
            can_quit_driver = True
        screen_shoter = get_shoter(screen_shoter, **shoter_kwargs)
        if qa_generator is not None:
            qa_generator = get_generator(qa_generator, **generator_kwargs)
        else:
            qa_generator = get_random_generator(**generator_kwargs)
        if checker is None:
            checker = get_random_generator(**generator_kwargs)
        else:
            checker = get_generator(checker, **generator_kwargs)
        if scorer is not None and isinstance(scorer, str):
            scorer = get_score_getter(scorer)
        elif scorer is None:
            scorer = get_random_score_getter()
        if question_finalizer is None:
            question_finalizer = QuestionFinalizer(**question_finalizer_kwargs)
        logs = []
        infomation_getter = InfomationExtractor()
        for website in tqdm(websites, desc="Capturing websites"):
            try:
                data, log = get_live_bench_data(driver, website, screen_shoter, qa_generator, checker, test=test, scorer=scorer, infomation_getter=infomation_getter, question_finalizer=question_finalizer)
                logs.append(log.to_dict())
                for d in data:
                    self.add(d)
            except Exception as e:
                logger.error(f"Error capturing website: {e}")
                logger.error(f"Website: {website.get_info()}")
                logs.append(
                    {
                        "success": False,
                        "content": f"Error capturing website: {e}",
                        "full_log": {
                            "website": website.get_info(),
                            "error": str(e),
                        },
                    }
                )
                continue
        if not os.path.exists(log_folder):
            os.makedirs(log_folder)
        date_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        log_file = os.path.join(log_folder, f"{date_time}.json")
        full_log = {
            "info": {
                "date_time": date_time,
                "screen_shoter": screen_shoter.get_name(),
                "qa_generator": qa_generator.get_name(),
                "checker": checker.get_name(),
                "scorer": scorer.get_name(),
            },
            "websites": [w.get_info() for w in websites],
            "logs": logs,
        }
        with open(log_file, "w") as f:
            json.dump(full_log, f, indent=4)
        logger.info(f"Logs saved to {os.path.abspath(log_file)}")
        if can_quit_driver:
            driver.quit()

    def upload(self, **kwargs):
        self.hf_data.push_to_hub(self.remote_path, config_name=self.name, split=self.split, **kwargs)

    def save(self, path: str):
        self.hf_data.save_to_disk(path)
        logger.info(f"Data saved to {os.path.abspath(path)}")
