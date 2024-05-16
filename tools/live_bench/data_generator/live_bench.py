import os
from typing import List
from tqdm import tqdm
from lmms_eval.live_bench.data_generator.live_bench_data import LiveBenchData
from datasets import Dataset, load_dataset
from lmms_eval.live_bench.websites import Website
from lmms_eval.live_bench.driver import load_driver
from lmms_eval.live_bench.data_generator import get_generator, get_random_generator
from lmms_eval.live_bench.screen_shoter import get_shoter
from lmms_eval.live_bench.data_generator.qa_generator import QAGenerator, QAData
from lmms_eval.live_bench.screen_shoter import ScreenImage, ScreenShoter
from lmms_eval.live_bench.data_generator.score_getter import get_score_getter, get_random_score_getter
import datasets

from typing import List
import logging

logger = logging.getLogger("lmms-eval")


def get_qa_data(images: ScreenImage, qa_generator: QAGenerator, test=False) -> List[QAData]:
    response = qa_generator.generate(images, test=test)
    qa_data = qa_generator.format_response(response)
    return qa_data


def get_live_bench_data(driver, website: Website, screen_shoter: ScreenShoter, qa_generator: QAGenerator, checker: QAGenerator, test=False, scorer=None) -> List[LiveBenchData]:
    images = screen_shoter.capture(driver, website)
    qa_data = get_qa_data(images, qa_generator, test=test)
    data = []
    for qa in qa_data:
        data.append(LiveBenchData(screen=images, question=qa.question, answer=qa.answer, subtask=qa.subtask, data_generator=qa_generator.get_name(), checker=checker, scorer=scorer))
    return data


class LiveBench(object):
    def __init__(self, path: str = "lmms-lab/LiveBench", *, name="default", split="test", cache_dir=None, remote_path=None, trust_remote_code=True, force_clear=False, **kwargs):
        self.path = path
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
                self.hf_data = load_dataset(self.path, name=name, split=split, cache_dir=cache_dir, trust_remote_code=trust_remote_code, **kwargs)
            except Exception as e:
                logger.error(f"Error loading dataset: {e}")
                self.clear()

    def clear(self):
        self.hf_data = Dataset.from_dict(
            {"id": [], "images": [], "website": [], "question": [], "answer": [], "subtask": [], "data_generator": [], "checker": [], "date_time": [], "screen_shoter": [], "screen_size": [], "score": [], "reason": [], "scorer_name": []},
            features=LiveBenchData.features,
        )

    def add(self, data: LiveBenchData, id: int = None):
        if id is None:
            id = len(self.hf_data)
        organized_data = data.to_hf_dict()
        organized_data["id"] = id
        self.hf_data = self.hf_data.add_item(organized_data)

    def capture(self, websites: List[Website] = None, *, screen_shoter="single_screen", qa_generator=None, checker=None, driver=None, scorer=None, test=False, driver_kwargs={}, shoter_kwargs={}, generator_kwargs={}):
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
        for website in tqdm(websites, desc="Capturing websites"):
            try:
                data = get_live_bench_data(driver, website, screen_shoter, qa_generator, checker, test=test, scorer=scorer)
                for d in data:
                    self.add(d)
            except Exception as e:
                logger.error(f"Error capturing website: {e}")
                logger.error(f"Website: {website.get_info()}")
                continue
        if can_quit_driver:
            driver.quit()

    def upload(self, **kwargs):
        self.hf_data.push_to_hub(self.remote_path, config_name=self.name, split=self.split, **kwargs)

    def save(self, path: str):
        self.hf_data.save_to_disk(path)
        logger.info(f"Data saved to {os.path.abspath(path)}")
