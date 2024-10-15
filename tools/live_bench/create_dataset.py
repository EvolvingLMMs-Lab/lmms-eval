from live_bench import LiveBench
from live_bench.websites import load_websites, load_websites_from_file

if __name__ == "__main__":
    website = load_websites()
    dataset = LiveBench(name="2024-09")

    website = load_websites_from_file("/data/pufanyi/project/lmms-eval/tools/temp/processed_images/selected")
    dataset.capture(websites=website, screen_shoter="human", qa_generator="claude", scorer="claude", checker="gpt4v", driver_kwargs={}, shoter_kwargs={}, generator_kwargs={})
    dataset.upload()
