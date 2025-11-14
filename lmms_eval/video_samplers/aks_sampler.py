from .base import BaseVideoSampler
from . import register_video_sampler
from typing import Any, Dict, Optional
import torchvision.transforms as T
from lavis.models import load_model_and_preprocess
from torchvision.transforms.functional import InterpolationMode
import numpy as np
import heapq
import torch
from PIL import Image


@register_video_sampler("aks")
class AKSVideoSampler(BaseVideoSampler):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = self.__class__.__name__
        self.extract_feature_model = kwargs.get("extract_feature_model", "blip")
        self.load_feature_model()
        self.max_num_frames = kwargs.get("max_num_frames", 64)
        self.ratio = kwargs.get("ratio", 1)
        self.t1 = kwargs.get("t1", 0.8)
        self.t2 = kwargs.get("t2", -100)
        self.all_depth = kwargs.get("all_depth", 5)

    def load_feature_model(self):
        if self.extract_feature_model == 'blip':
            self.model, self.vis_processors, self.text_processors = load_model_and_preprocess("blip_image_text_matching", "large", device=self.device, is_eval=True)
            self.vis_processors['eval_tensor'] = self.compose_to_tensor_transform(self.vis_processors["eval"].transform)
        elif self.extract_feature_model == 'clip':
            from transformers import CLIPModel, CLIPProcessor
            self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.model.to(self.device)
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        elif self.extract_feature_model == 'sevila':
            self.model, self.vis_processors, self.text_processors = load_model_and_preprocess(name="sevila", model_type="pretrain_flant5xl", is_eval=True, device=self.device)
            self.vis_processors['eval_tensor'] = self.compose_to_tensor_transform(self.vis_processors["eval"].transform)
        else:
            raise ValueError(f"model {extract_feature_model} not supported")
    
    def meanstd(self, len_scores, dic_scores, n, fns,t1,t2,all_depth):
        split_scores = []
        split_fn = []
        no_split_scores = []
        no_split_fn = []
        i= 0
        for dic_score, fn in zip(dic_scores, fns):
                # normalized_data = (score - np.min(score)) / (np.max(score) - np.min(score))
                score = dic_score['score']
                depth = dic_score['depth']
                mean = np.mean(score)
                std = np.std(score)

                top_n = heapq.nlargest(n, range(len(score)), score.__getitem__)
                top_score = [score[t] for t in top_n]
                # print(f"split {i}: ",len(score))
                i += 1
                mean_diff = np.mean(top_score) - mean
                if mean_diff > t1 and std > t2:
                        no_split_scores.append(dic_score)
                        no_split_fn.append(fn)
                elif depth < all_depth:
                # elif len(score)>(len_scores/n)*2 and len(score) >= 8:
                        score1 = score[:len(score)//2]
                        score2 = score[len(score)//2:]
                        fn1 = fn[:len(score)//2]
                        fn2 = fn[len(score)//2:]                       
                        split_scores.append(dict(score=score1,depth=depth+1))
                        split_scores.append(dict(score=score2,depth=depth+1))
                        split_fn.append(fn1)
                        split_fn.append(fn2)
                else:
                        no_split_scores.append(dic_score)
                        no_split_fn.append(fn)
        if len(split_scores) > 0:
                all_split_score, all_split_fn = self.meanstd(len_scores, split_scores, n, split_fn,t1,t2,all_depth)
        else:
                all_split_score = []
                all_split_fn = []
        all_split_score = no_split_scores + all_split_score
        all_split_fn = no_split_fn + all_split_fn


        return all_split_score, all_split_fn

    def compose_to_tensor_transform(self, pil_compose: T.Compose) -> T.Compose:
        """
        Convert a PIL-based torchvision Compose (e.g., from LAVIS vis_processors['eval'])
        into a tensor-native Compose. Common ops (Resize/CenterCrop/Normalize/ToTensor)
        are translated to their tensor-friendly counterparts.

        Input expectation:
        - Tensor images as (C,H,W) or (B,C,H,W), dtype uint8 or float.
        - RGB order; handle permutes before calling if needed.

        Returns:
        - A torchvision.transforms.Compose that works on torch.Tensors.
        """
        mapped = []
        for t in pil_compose.transforms:
            name = t.__class__.__name__

            if name == "Resize":
                mapped.append(T.Resize(
                    size=t.size,
                    interpolation=getattr(t, "interpolation", InterpolationMode.BILINEAR),
                    antialias=getattr(t, "antialias", True),
                ))

            elif name == "CenterCrop":
                mapped.append(T.CenterCrop(size=t.size))

            elif name == "RandomResizedCrop":
                mapped.append(T.RandomResizedCrop(
                    size=t.size,
                    scale=t.scale,
                    ratio=t.ratio,
                    interpolation=getattr(t, "interpolation", InterpolationMode.BILINEAR),
                    antialias=getattr(t, "antialias", True),
                ))

            elif name == "ToTensor":
                # For tensor input, we only need dtype/scale (PIL->Tensor step not needed).
                mapped.append(T.ConvertImageDtype(torch.float32))

            elif name == "Normalize":
                mapped.append(T.Normalize(mean=t.mean, std=t.std, inplace=getattr(t, "inplace", False)))

            else:
                # Keep unknown transforms as-is (may still expect PIL).
                mapped.append(t)

        return T.Compose(mapped)

    def get_frames(self, vr, frame_num, backend):
        if backend == 'torchcodec':
            full_raw_image_tensors = vr.get_frames_at(indices=frame_num).data
        elif backend == 'decord':
            full_raw_image_tensors = torch.from_numpy(vr.get_batch(frame_num).asnumpy().permute(0, 3, 1, 2))
        elif backend == 'torchvision':
            full_raw_image_tensors = vr[frame_num]
        else:
            raise ValueError(f"backend {backend} not supported")
        return full_raw_image_tensors

    def sample(self, ele: Any, **kwargs) -> Optional[Dict[str, Any]]:
        # TODO: Implement AKS sampling
        video_path = ele["video"]
        text = ele['question']  
        vr = ele['video_reader']
        fps = ele['video_fps']
        frame_nums = int(ele["total_frames"]/int(fps))
        frame_num = [j*int(fps) for j in range(frame_nums)]
        score = []

        if self.extract_feature_model == 'blip':
            txt = self.text_processors["eval"](text)
            with torch.no_grad():
                for i in range(0, len(frame_num), self.batch_size):
                    batch = frame_num[i:i+self.batch_size]
                    full_raw_image_tensors = self.get_frames(vr, batch, ele['video_reader_backend'])
                    imgs = self.vis_processors['eval_tensor'](full_raw_image_tensors).to(self.device)
                    blip_output = self.model({"image": imgs, "text_input": [txt]*imgs.shape[0]}, match_head="itm")
                    blip_scores = torch.nn.functional.softmax(blip_output, dim=1)
                    score.extend(blip_scores[:, 1].tolist())
        elif self.extract_feature_model == 'clip':
            inputs_text = self.processor(text=text, return_tensors="pt", padding=True,truncation=True).to(self.device)
            text_features = self.model.get_text_features(**inputs_text)
            for j in range(frame_nums):
                raw_image_tensor = vr[j*int(fps)] 
                if ele['video_reader_backend'] != 'decord': 
                    raw_image_tensor = raw_image_tensor.permute(1,2,0)
                raw_image = np.array(raw_image_tensor.cpu())
                raw_image = Image.fromarray(raw_image)
                inputs_image = self.processor(images=raw_image, return_tensors="pt", padding=True).to(self.device)
                with torch.no_grad():
                    image_features = self.model.get_image_features(**inputs_image)
                clip_score = torch.nn.CosineSimilarity(dim=-1)(text_features, image_features)
                score.append(clip_score.item())
                frame_num.append(j*int(fps))

        elif self.extract_feature_model == 'sevila':
            text = 'Question: ' + data['question'] + ' Candidate: ' 
            for j,cad in enumerate(data['answer_choices']):
                text = text + ". ".join([chr(ord("A")+j), cad]) + ' '
            text = text + '. Is this a good frame that can answer the question?'
            txt = self.text_processors["eval"](text)
            full_raw_image_tensors = self.get_frames(vr, frame_num, ele['video_reader_backend'])
            with torch.no_grad():
                for batch in full_raw_image_tensors:
                    imgs = self.vis_processors['eval_tensor'](batch).unsqueeze(1).to(self.device)
                    samples = {'video':imgs,'loc_input':[txt]*imgs.shape[0]}
                    sevila_score = self.model.generate_score(samples).squeeze(1)
                    score.append(sevila_score)
            score = torch.cat(score, dim=0).detach().cpu().numpy()
        else:
            raise ValueError(f"model {self.extract_feature_model} not supported")
        
        nums = int(len(score)/self.ratio)
        new_score = [score[num*self.ratio] for num in range(nums)]
        new_fnum = [frame_num[num*self.ratio] for num in range(nums)]
        score = new_score
        fn = new_fnum
        num = self.max_num_frames
        if len(score) >= num:
            normalized_data = (score - np.min(score)) / (np.max(score) - np.min(score))
            a, b = self.meanstd(len(score), [dict(score=normalized_data,depth=0)], num, [fn], self.t1, self.t2, self.all_depth)
            out = []
            if len(score) >= num:
                for s,f in zip(a,b): 
                    f_num = int(num / 2**(s['depth']))
                    topk = heapq.nlargest(f_num, range(len(s['score'])), s['score'].__getitem__)
                    f_nums = [f[t] for t in topk]
                    out.extend(f_nums)
            out.sort()
            return out
        else:
            return fn