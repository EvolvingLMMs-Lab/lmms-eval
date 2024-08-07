from live_bench.websites import load_websites, load_websites_from_file
from live_bench import LiveBench


if __name__ == "__main__":
    website = load_websites()
    dataset = LiveBench()
    dataset.capture(websites=website, driver_kwargs={"headless": True}, screen_shoter="single_screen", shoter_kwargs={"screen_size": (1024, 1024)}, qa_generator="gpt4v", scorer="claude", checker="gemini")

    website = load_websites_from_file("/data/pufanyi/project/lmms-eval/temp/images")
    dataset.capture(websites=website, screen_shoter="human", qa_generator="gpt4v", scorer="claude", checker="gemini", driver_kwargs={}, shoter_kwargs={}, generator_kwargs={})
    dataset.upload()
