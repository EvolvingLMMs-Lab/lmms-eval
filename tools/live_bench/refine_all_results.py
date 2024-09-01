from datasets import Dataset, load_dataset
from live_bench.data_generator.question_finalizer import QuestionFinalizer
from tqdm import tqdm

if __name__ == "__main__":
    hf_data = load_dataset("lmms-lab/LiveBench", "2024-07", split="test")
    finalizer = QuestionFinalizer()

    def load_results():
        for item in tqdm(hf_data):
            try:
                res = finalizer.finalize_question(question=item["question"], answer=item["answer"], criteria=item["criteria"], images=item["images"])
                final_answer = item.copy()
                final_answer["question"] = res["question"]
                final_answer["answer"] = res["answer"]
                final_answer["criteria"] = res["criteria"]
                print(item)
                print(final_answer)
            except Exception as e:
                print(f"Error in {item['id']}: {e}")
                final_answer = item

            yield final_answer
            # break

    final_data = {}
    for data in load_results():
        for item, value in data.items():
            if item not in final_data:
                final_data[item] = []
            final_data[item].append(value)
    # final_data = Dataset.from_generator(load_results)
    final_data = Dataset.from_dict(final_data, features=hf_data.features)
    final_data.save_to_disk("logs/2024-07-final")
    final_data.push_to_hub("lmms-lab/LiveBench", "2024-07")
