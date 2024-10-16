import os
from random import sample

import yaml
from live_bench.websites.website import DefaultWebsite, HumanScreenShotWebsite, Website


def get_website(website_dict):
    if "website_class" not in website_dict:
        website_class = DefaultWebsite
    else:
        website_class = website_dict["website_class"]
    url = website_dict["url"]
    if "args" in website_dict:
        return website_class(url, **website_dict["args"])
    else:
        return website_class(url)


def load_websites(num_sample: int = -1):
    website_list_path = os.path.join(os.path.dirname(__file__), "website_list.yaml")
    with open(website_list_path, "r") as f:
        website_list = yaml.full_load(f)["websites"]
    if num_sample > 0:
        website_list = sample(website_list, num_sample)
    return [get_website(website_dict) for website_dict in website_list]


def load_websites_from_file(file_path):
    names = os.listdir(file_path)
    websites = []
    for name in names:
        path = os.path.join(file_path, name)
        websites.append(HumanScreenShotWebsite(path=path, name=name))
    return websites
