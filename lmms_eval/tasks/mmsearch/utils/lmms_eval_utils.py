import json
import os


def save_result_to_cache(doc, round_res, previous_round_info, save_dir):
    save_dict = dict(
        sample_id=doc["sample_id"],
        query=doc["query"],
        round_res=round_res,
    )
    save_dict.update(previous_round_info)
    json.dump(save_dict, open(os.path.join(save_dir, f"{save_dict['sample_id']}.json"), "w"), indent=4)
