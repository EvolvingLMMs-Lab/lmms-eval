import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer


CHAIR_METRICS = ["chair_s", "chair_i"]

MSCOCO_OBJECTS = ['person', 'girl', 'boy', 'man', 'woman', 'kid', 'child', 'chef', 'baker', 'people', 'adult', 'rider', 'children', 'baby', 'worker', 'passenger', 'sister', 'biker', 'policeman', 'cop', 'officer', 'lady', 'cowboy', 'bride', 'groom', 'male', 'female', 'guy', 'traveler', 'mother', 'father', 'gentleman', 'pitcher', 'player', 'skier', 'snowboarder', 'skater', 'skateboarder', 'person', 'woman', 'guy', 'foreigner', 'child', 'gentleman', 'caller', 'offender', 'coworker', 'trespasser', 'patient', 'politician', 'soldier', 'grandchild', 'serviceman', 'walker', 'drinker', 'doctor', 'bicyclist', 'thief', 'buyer', 'teenager', 'student', 'camper', 'driver', 'solider', 'hunter', 'shopper', 'villager', 'bicycle', 'bike', 'bicycle', 'bike', 'unicycle', 'minibike', 'trike', 'car', 'automobile', 'van', 'minivan', 'sedan', 'suv', 'hatchback', 'cab', 'jeep', 'coupe', 'taxicab', 'limo', 'taxi', 'motorcycle', 'scooter', ' motor bike', 'motor cycle', 'motorbike', 'scooter', 'moped', 'airplane', 'jetliner', 'plane', 'air plane', 'monoplane', 'aircraft', 'jet', 'jetliner', 'airbus', 'biplane', 'seaplane', 'bus', 'minibus', 'trolley', 'train', 'locomotive', 'tramway', 'caboose', 'truck', 'pickup', 'lorry', 'hauler', 'firetruck', 'boat', 'ship', 'liner', 'sailboat', 'motorboat', 'dinghy', 'powerboat', 'speedboat', 'canoe', 'skiff', 'yacht', 'kayak', 'catamaran', 'pontoon', 'houseboat', 'vessel', 'rowboat', 'trawler', 'ferryboat', 'watercraft', 'tugboat', 'schooner', 'barge', 'ferry', 'sailboard', 'paddleboat', 'lifeboat', 'freighter', 'steamboat', 'riverboat', 'battleship', 'steamship', 'traffic light', 'street light', 'traffic signal', 'stop light', 'streetlight', 'stoplight', 'fire hydrant', 'hydrant', 'stop sign', 'parking meter', 'bench', 'pew', 'bird', 'ostrich', 'owl', 'seagull', 'goose', 'duck', 'parakeet', 'falcon', 'robin', 'pelican', 'waterfowl', 'heron', 'hummingbird', 'mallard', 'finch', 'pigeon', 'sparrow', 'seabird', 'osprey', 'blackbird', 'fowl', 'shorebird', 'woodpecker', 'egret', 'chickadee', 'quail', 'bluebird', 'kingfisher', 'buzzard', 'willet', 'gull', 'swan', 'bluejay', 'flamingo', 'cormorant', 'parrot', 'loon', 'gosling', 'waterbird', 'pheasant', 'rooster', 'sandpiper', 'crow', 'raven', 'turkey', 'oriole', 'cowbird', 'warbler', 'magpie', 'peacock', 'cockatiel', 'lorikeet', 'puffin', 'vulture', 'condor', 'macaw', 'peafowl', 'cockatoo', 'songbird', 'cat', 'kitten', 'feline', 'tabby', 'dog', 'puppy', 'beagle', 'pup', 'chihuahua', 'schnauzer', 'dachshund', 'rottweiler', 'canine', 'pitbull', 'collie', 'pug', 'terrier', 'poodle', 'labrador', 'doggie', 'doberman', 'mutt', 'doggy', 'spaniel', 'bulldog', 'sheepdog', 'weimaraner', 'corgi', 'cocker', 'greyhound', 'retriever', 'brindle', 'hound', 'whippet', 'husky', 'horse', 'colt', 'pony', 'racehorse', 'stallion', 'equine', 'mare', 'foal', 'palomino', 'mustang', 'clydesdale', 'bronc', 'bronco', 'sheep', 'lamb', 'ram', 'lamb', 'goat', 'ewe', 'cow', 'cattle', 'oxen', 'ox', 'calf', 'cattle', 'holstein', 'heifer', 'buffalo', 'bull', 'zebu', 'bison', 'elephant', 'bear', 'panda', 'zebra', 'giraffe', 'backpack', 'knapsack', 'umbrella', 'handbag', 'wallet', 'purse', 'briefcase', 'tie', 'bow', 'bow tie', 'suitcase', 'suit case', 'luggage', 'frisbee', 'skis', 'ski', 'snowboard', 'sports ball', 'ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'longboard', 'skimboard', 'shortboard', 'wakeboard', 'tennis racket', 'racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'pocketknife', 'knive', 'spoon', 'bowl', 'container', 'banana', 'apple', 'sandwich', 'burger', 'sub', 'cheeseburger', 'hamburger', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'doughnut', 'bagel', 'cake', ' cheesecake', 'cupcake', 'shortcake', 'coffeecake', 'pancake', 'chair', 'seat', 'stool', 'couch', 'sofa', 'recliner', 'futon', 'loveseat', 'settee', 'chesterfield', 'potted plant', 'houseplant', 'bed', 'dining table', 'table', 'desk', 'toilet', 'urinal', 'commode', 'toilet', 'lavatory', 'potty', 'tv', 'monitor', 'televison', 'television', 'laptop', 'computer', 'notebook', 'netbook', 'lenovo', 'macbook', 'laptop computer', 'mouse', 'remote', 'keyboard', 'cell phone', 'mobile phone', 'phone', 'cellphone', 'telephone', 'phon', 'smartphone', 'iPhone', 'microwave', 'oven', 'stovetop', 'stove', 'stove top oven', 'toaster', 'sink', 'refrigerator', 'fridge', 'fridge', 'freezer', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'teddybear', 'hair drier', 'hairdryer', 'toothbrush']
INVERSE_SYNONYM_DICT = {'person': 'person', 'girl': 'person', 'boy': 'person', 'man': 'person', 'woman': 'person', 'kid': 'person', 'child': 'person', 'chef': 'person', 'baker': 'person', 'people': 'person', 'adult': 'person', 'rider': 'person', 'children': 'person', 'baby': 'person', 'worker': 'person', 'passenger': 'person', 'sister': 'person', 'biker': 'person', 'policeman': 'person', 'cop': 'person', 'officer': 'person', 'lady': 'person', 'cowboy': 'person', 'bride': 'person', 'groom': 'person', 'male': 'person', 'female': 'person', 'guy': 'person', 'traveler': 'person', 'mother': 'person', 'father': 'person', 'gentleman': 'person', 'pitcher': 'person', 'player': 'person', 'skier': 'person', 'snowboarder': 'person', 'skater': 'person', 'skateboarder': 'person', 'foreigner': 'person', 'caller': 'person', 'offender': 'person', 'coworker': 'person', 'trespasser': 'person', 'patient': 'person', 'politician': 'person', 'soldier': 'person', 'grandchild': 'person', 'serviceman': 'person', 'walker': 'person', 'drinker': 'person', 'doctor': 'person', 'bicyclist': 'person', 'thief': 'person', 'buyer': 'person', 'teenager': 'person', 'student': 'person', 'camper': 'person', 'driver': 'person', 'solider': 'person', 'hunter': 'person', 'shopper': 'person', 'villager': 'person', 'bicycle': 'bicycle', 'bike': 'bicycle', 'unicycle': 'bicycle', 'minibike': 'bicycle', 'trike': 'bicycle', 'car': 'car', 'automobile': 'car', 'van': 'car', 'minivan': 'car', 'sedan': 'car', 'suv': 'car', 'hatchback': 'car', 'cab': 'car', 'jeep': 'car', 'coupe': 'car', 'taxicab': 'car', 'limo': 'car', 'taxi': 'car', 'motorcycle': 'motorcycle', 'scooter': 'motorcycle', ' motor bike': 'motorcycle', 'motor cycle': 'motorcycle', 'motorbike': 'motorcycle', 'moped': 'motorcycle', 'airplane': 'airplane', 'jetliner': 'airplane', 'plane': 'airplane', 'air plane': 'airplane', 'monoplane': 'airplane', 'aircraft': 'airplane', 'jet': 'airplane', 'airbus': 'airplane', 'biplane': 'airplane', 'seaplane': 'airplane', 'bus': 'bus', 'minibus': 'bus', 'trolley': 'bus', 'train': 'train', 'locomotive': 'train', 'tramway': 'train', 'caboose': 'train', 'truck': 'truck', 'pickup': 'truck', 'lorry': 'truck', 'hauler': 'truck', 'firetruck': 'truck', 'boat': 'boat', 'ship': 'boat', 'liner': 'boat', 'sailboat': 'boat', 'motorboat': 'boat', 'dinghy': 'boat', 'powerboat': 'boat', 'speedboat': 'boat', 'canoe': 'boat', 'skiff': 'boat', 'yacht': 'boat', 'kayak': 'boat', 'catamaran': 'boat', 'pontoon': 'boat', 'houseboat': 'boat', 'vessel': 'boat', 'rowboat': 'boat', 'trawler': 'boat', 'ferryboat': 'boat', 'watercraft': 'boat', 'tugboat': 'boat', 'schooner': 'boat', 'barge': 'boat', 'ferry': 'boat', 'sailboard': 'boat', 'paddleboat': 'boat', 'lifeboat': 'boat', 'freighter': 'boat', 'steamboat': 'boat', 'riverboat': 'boat', 'battleship': 'boat', 'steamship': 'boat', 'traffic light': 'traffic light', 'street light': 'traffic light', 'traffic signal': 'traffic light', 'stop light': 'traffic light', 'streetlight': 'traffic light', 'stoplight': 'traffic light', 'fire hydrant': 'fire hydrant', 'hydrant': 'fire hydrant', 'stop sign': 'stop sign', 'parking meter': 'parking meter', 'bench': 'bench', 'pew': 'bench', 'bird': 'bird', 'ostrich': 'bird', 'owl': 'bird', 'seagull': 'bird', 'goose': 'bird', 'duck': 'bird', 'parakeet': 'bird', 'falcon': 'bird', 'robin': 'bird', 'pelican': 'bird', 'waterfowl': 'bird', 'heron': 'bird', 'hummingbird': 'bird', 'mallard': 'bird', 'finch': 'bird', 'pigeon': 'bird', 'sparrow': 'bird', 'seabird': 'bird', 'osprey': 'bird', 'blackbird': 'bird', 'fowl': 'bird', 'shorebird': 'bird', 'woodpecker': 'bird', 'egret': 'bird', 'chickadee': 'bird', 'quail': 'bird', 'bluebird': 'bird', 'kingfisher': 'bird', 'buzzard': 'bird', 'willet': 'bird', 'gull': 'bird', 'swan': 'bird', 'bluejay': 'bird', 'flamingo': 'bird', 'cormorant': 'bird', 'parrot': 'bird', 'loon': 'bird', 'gosling': 'bird', 'waterbird': 'bird', 'pheasant': 'bird', 'rooster': 'bird', 'sandpiper': 'bird', 'crow': 'bird', 'raven': 'bird', 'turkey': 'bird', 'oriole': 'bird', 'cowbird': 'bird', 'warbler': 'bird', 'magpie': 'bird', 'peacock': 'bird', 'cockatiel': 'bird', 'lorikeet': 'bird', 'puffin': 'bird', 'vulture': 'bird', 'condor': 'bird', 'macaw': 'bird', 'peafowl': 'bird', 'cockatoo': 'bird', 'songbird': 'bird', 'cat': 'cat', 'kitten': 'cat', 'feline': 'cat', 'tabby': 'cat', 'dog': 'dog', 'puppy': 'dog', 'beagle': 'dog', 'pup': 'dog', 'chihuahua': 'dog', 'schnauzer': 'dog', 'dachshund': 'dog', 'rottweiler': 'dog', 'canine': 'dog', 'pitbull': 'dog', 'collie': 'dog', 'pug': 'dog', 'terrier': 'dog', 'poodle': 'dog', 'labrador': 'dog', 'doggie': 'dog', 'doberman': 'dog', 'mutt': 'dog', 'doggy': 'dog', 'spaniel': 'dog', 'bulldog': 'dog', 'sheepdog': 'dog', 'weimaraner': 'dog', 'corgi': 'dog', 'cocker': 'dog', 'greyhound': 'dog', 'retriever': 'dog', 'brindle': 'dog', 'hound': 'dog', 'whippet': 'dog', 'husky': 'dog', 'horse': 'horse', 'colt': 'horse', 'pony': 'horse', 'racehorse': 'horse', 'stallion': 'horse', 'equine': 'horse', 'mare': 'horse', 'foal': 'horse', 'palomino': 'horse', 'mustang': 'horse', 'clydesdale': 'horse', 'bronc': 'horse', 'bronco': 'horse', 'sheep': 'sheep', 'lamb': 'sheep', 'ram': 'sheep', 'goat': 'sheep', 'ewe': 'sheep', 'cow': 'cow', 'cattle': 'cow', 'oxen': 'cow', 'ox': 'cow', 'calf': 'cow', 'holstein': 'cow', 'heifer': 'cow', 'buffalo': 'cow', 'bull': 'cow', 'zebu': 'cow', 'bison': 'cow', 'elephant': 'elephant', 'bear': 'bear', 'panda': 'bear', 'zebra': 'zebra', 'giraffe': 'giraffe', 'backpack': 'backpack', 'knapsack': 'backpack', 'umbrella': 'umbrella', 'handbag': 'handbag', 'wallet': 'handbag', 'purse': 'handbag', 'briefcase': 'handbag', 'tie': 'tie', 'bow': 'tie', 'bow tie': 'tie', 'suitcase': 'suitcase', 'suit case': 'suitcase', 'luggage': 'suitcase', 'frisbee': 'frisbee', 'skis': 'skis', 'ski': 'skis', 'snowboard': 'snowboard', 'sports ball': 'sports ball', 'ball': 'sports ball', 'kite': 'kite', 'baseball bat': 'baseball bat', 'baseball glove': 'baseball glove', 'skateboard': 'skateboard', 'surfboard': 'surfboard', 'longboard': 'surfboard', 'skimboard': 'surfboard', 'shortboard': 'surfboard', 'wakeboard': 'surfboard', 'tennis racket': 'tennis racket', 'racket': 'tennis racket', 'bottle': 'bottle', 'wine glass': 'wine glass', 'cup': 'cup', 'fork': 'fork', 'knife': 'knife', 'pocketknife': 'knife', 'knive': 'knife', 'spoon': 'spoon', 'bowl': 'bowl', 'container': 'bowl', 'banana': 'banana', 'apple': 'apple', 'sandwich': 'sandwich', 'burger': 'sandwich', 'sub': 'sandwich', 'cheeseburger': 'sandwich', 'hamburger': 'sandwich', 'orange': 'orange', 'broccoli': 'broccoli', 'carrot': 'carrot', 'hot dog': 'hot dog', 'pizza': 'pizza', 'donut': 'donut', 'doughnut': 'donut', 'bagel': 'donut', 'cake': 'cake', ' cheesecake': 'cake', 'cupcake': 'cake', 'shortcake': 'cake', 'coffeecake': 'cake', 'pancake': 'cake', 'chair': 'chair', 'seat': 'chair', 'stool': 'chair', 'couch': 'couch', 'sofa': 'couch', 'recliner': 'couch', 'futon': 'couch', 'loveseat': 'couch', 'settee': 'couch', 'chesterfield': 'couch', 'potted plant': 'potted plant', 'houseplant': 'potted plant', 'bed': 'bed', 'dining table': 'dining table', 'table': 'dining table', 'desk': 'dining table', 'toilet': 'toilet', 'urinal': 'toilet', 'commode': 'toilet', 'lavatory': 'toilet', 'potty': 'toilet', 'tv': 'tv', 'monitor': 'tv', 'televison': 'tv', 'television': 'tv', 'laptop': 'laptop', 'computer': 'laptop', 'notebook': 'laptop', 'netbook': 'laptop', 'lenovo': 'laptop', 'macbook': 'laptop', 'laptop computer': 'laptop', 'mouse': 'mouse', 'remote': 'remote', 'keyboard': 'keyboard', 'cell phone': 'cell phone', 'mobile phone': 'cell phone', 'phone': 'cell phone', 'cellphone': 'cell phone', 'telephone': 'cell phone', 'phon': 'cell phone', 'smartphone': 'cell phone', 'iPhone': 'cell phone', 'microwave': 'microwave', 'oven': 'oven', 'stovetop': 'oven', 'stove': 'oven', 'stove top oven': 'oven', 'toaster': 'toaster', 'sink': 'sink', 'refrigerator': 'refrigerator', 'fridge': 'refrigerator', 'freezer': 'refrigerator', 'book': 'book', 'clock': 'clock', 'vase': 'vase', 'scissors': 'scissors', 'teddy bear': 'teddy bear', 'teddybear': 'teddy bear', 'hair drier': 'hair drier', 'hairdryer': 'hair drier', 'toothbrush': 'toothbrush'}
DOUBLE_WORD_DICT = {'motor bike': 'motor bike', 'motor cycle': 'motor cycle', 'air plane': 'air plane', 'traffic light': 'traffic light', 'street light': 'street light', 'traffic signal': 'traffic signal', 'stop light': 'stop light', 'fire hydrant': 'fire hydrant', 'stop sign': 'stop sign', 'parking meter': 'parking meter', 'suit case': 'suit case', 'sports ball': 'sports ball', 'baseball bat': 'baseball bat', 'baseball glove': 'baseball glove', 'tennis racket': 'tennis racket', 'wine glass': 'wine glass', 'hot dog': 'hot dog', 'cell phone': 'cell phone', 'mobile phone': 'mobile phone', 'teddy bear': 'teddy bear', 'hair drier': 'hair drier', 'potted plant': 'potted plant', 'bow tie': 'tie', 'laptop computer': 'laptop computer', 'stove top oven': 'stove top oven', 'home plate': 'home plate', 'train track': 'train track', 'baby bird': 'bird', 'adult bird': 'bird', 'baby cat': 'cat', 'adult cat': 'cat', 'baby dog': 'dog', 'adult dog': 'dog', 'baby horse': 'horse', 'adult horse': 'horse', 'baby sheep': 'sheep', 'adult sheep': 'sheep', 'baby cow': 'cow', 'adult cow': 'cow', 'baby elephant': 'elephant', 'adult elephant': 'elephant', 'baby bear': 'bear', 'adult bear': 'bear', 'baby zebra': 'zebra', 'adult zebra': 'zebra', 'baby giraffe': 'giraffe', 'adult giraffe': 'giraffe', 'baby animal': 'animal', 'adult animal': 'animal', 'baby cub': 'cub', 'adult cub': 'cub', 'passenger jet': 'jet', 'passenger train': 'train', 'toilet seat': 'toilet', 'wine glas': 'wine glass'}

def coco_cap_chair_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def coco_cap_chair_doc_to_text(doc):
    return f"Please describe this image in detail."


def caption_to_words(caption):
    
    '''
    Input: caption
    Output: MSCOCO words in the caption
    '''

    def get_wordnet_pos(tag):
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            return None

    #standard preprocessing
    words = nltk.word_tokenize(caption.lower())
    tagged_sent = nltk.pos_tag(words)
    lemmas_sent = []
    wnl = WordNetLemmatizer()

    for tag in tagged_sent:
        wordnet_pos = get_wordnet_pos(tag[1]) or wordnet.NOUN
        lemmas_sent.append(wnl.lemmatize(tag[0], pos=wordnet_pos))
    # words = [singularize(w) for w in words]
    words = lemmas_sent

    #replace double words
    i = 0
    double_words = []
    idxs = []
    while i < len(words):
        idxs.append(i) 
        double_word = ' '.join(words[i:i+2])
        if double_word in DOUBLE_WORD_DICT: 
            double_words.append(DOUBLE_WORD_DICT[double_word])
            i += 2
        else:
            double_words.append(words[i])
            i += 1
    words = double_words

    #toilet seat is not chair (sentences like "the seat of the toilet" will fire for "chair" if we do not include this line)
    if ('toilet' in words) & ('seat' in words): words = [word for word in words if word != 'seat']

    #get synonyms for all words in the caption
    idxs = [idxs[idx] for idx, word in enumerate(words) \
            if word in set(MSCOCO_OBJECTS)]
    words = [word for word in words if word in set(MSCOCO_OBJECTS)]
    node_words = []
    for word in words:
        node_words.append(INVERSE_SYNONYM_DICT[word])
    #return all the MSCOCO objects in the caption
    return words, node_words, idxs, double_words


def coco_cap_chair_process_result(doc, result):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name, value: metric value
    """
    pred = result[0] if len(result) > 0 else ""
    words, node_words, idxs, raw_words = caption_to_words(pred)
    image_id = int(doc["question_id"])

    data_dict = {"answer": doc["gt_object"], "pred": node_words, "image_id": image_id}

    return {f"coco_cap_{metric}": data_dict for metric in CHAIR_METRICS}


def coco_cap_chair_aggregate_results_chair_i(results):
    """
    Args:
        results: a list of values returned by process_results
    Returns:
        A score
    """
    num_all_mentioned_objects = 0
    num_hallucinated_objects = 0
    for result in results:
        gt_object = result["answer"]
        pred = result["pred"]
        num_all_mentioned_objects += len(pred)
        # calculate the number of hallucination
        for node_word in pred:
            if node_word not in gt_object:
                num_hallucinated_objects += 1
    return (num_hallucinated_objects / num_all_mentioned_objects) * 100


def coco_cap_chair_aggregate_results_chair_s(results):
    """
    Args:
        results: a list of values returned by process_results
    Returns:
        A score
    """
    num_samples = len(results)
    num_hallucinated_samples = 0
    for result in results:
        gt_object = result["answer"]
        pred = result["pred"]
        # calculate the number of hallucination
        for node_word in pred:
            if node_word not in gt_object:
                num_hallucinated_samples += 1
                break
    return (num_hallucinated_samples / num_samples) * 100

            