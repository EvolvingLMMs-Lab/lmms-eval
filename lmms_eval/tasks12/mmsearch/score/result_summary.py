def get_area_score(prediction_summary, key):
    area_pre_dict = {"news": [], "knowledge": []}
    for inst in prediction_summary:
        area = inst["area"]
        area_pre_dict[area].append(inst[key])
    area_dict = dict()
    for k, v in area_pre_dict.items():
        area_dict[k] = dict(length=len(v), average=sum(v) / len(v))
    return area_dict


def get_subfield_score(prediction_summary, key, all_subfield):
    area_pre_dict = {t: [] for t in all_subfield}
    for inst in prediction_summary:
        area = inst["subfield"]
        area_pre_dict[area].append(inst[key])
    area_dict = dict()
    for k, v in area_pre_dict.items():
        area_dict[k] = dict(length=len(v), average=sum(v) / len(v))
    return area_dict


def get_result_summary(anno, result_list, summary_key):
    if isinstance(summary_key, str):
        summary_key = [summary_key]
    # change result_list to dict
    result_dict = {inst["sample_id"]: inst for inst in result_list}

    all_subfield = []
    # add missing samples to zero
    for inst in anno:
        if inst["sample_id"] not in result_dict:
            dummy_result = dict(sample_id=inst["sample_id"], area=inst["area"], subfield=inst["subfield"], **{k: 0 for k in summary_key})
            result_list.append(dummy_result)
            print(f"Missing sample: {inst['sample_id']}")
        all_subfield.append(inst["subfield"])
    all_subfield = list(set(all_subfield))

    return_dict = dict()
    for key in summary_key:
        try:
            return_dict[key] = dict(
                total_dict=dict(total_length=len(result_list), average=sum([inst[key] for inst in result_list]) / len(result_list)),
                area_dict=get_area_score(result_list, key),
                subfield_dict=get_subfield_score(result_list, key, all_subfield),
            )
        except:
            import pdb

            pdb.set_trace()
            print(result_dict)

    return return_dict
