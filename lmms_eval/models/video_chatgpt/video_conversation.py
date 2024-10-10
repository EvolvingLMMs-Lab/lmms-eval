import dataclasses
from enum import Enum, auto
from typing import List

from lmms_eval.models.video_chatgpt.eval.model_utils import load_video


class SeparatorStyle(Enum):
    """Different separator style."""

    SINGLE = auto()
    TWO = auto()
    MPT = auto()


@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""

    system: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE
    sep: str = "###"
    sep2: str = None
    version: str = "Unknown"

    skip_next: bool = False

    def get_prompt(self):
        if self.sep_style == SeparatorStyle.SINGLE:
            ret = self.system + self.sep
            for role, message in self.messages:
                if message:
                    if type(message) is tuple:
                        message, _ = message
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.TWO:
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message:
                    if type(message) is tuple:
                        message, _ = message
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
            return ret
        if self.sep_style == SeparatorStyle.MPT:
            ret = self.system + self.sep
            for role, message in self.messages:
                if message:
                    if type(message) is tuple:
                        message, _ = message
                    ret += role + message + self.sep
                else:
                    ret += role
            return ret
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

    def append_message(self, role, message):
        self.messages.append([role, message])

    def get_video_frames(self, n_clips=1, num_frm=100):
        video_frames = []
        for i, (role, msg) in enumerate(self.messages[self.offset :]):
            if i % 2 == 0:
                if type(msg) is tuple:
                    msg, video_path = msg

                    clip_imgs = load_video(video_path, n_clips, num_frm)

                    for image in clip_imgs:
                        video_frames.append(image)
        return video_frames

    def to_gradio_chatbot(self):
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset :]):
            if i % 2 == 0:
                if type(msg) is tuple:
                    msg, image = msg
                ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        # Hack to make the demo work
        try:
            if "<video>" in ret[0][0]:
                ret[0][0] = ret[0][0].replace("<video>", "")
        except Exception as e:
            pass

        return ret

    def copy(self):
        return Conversation(system=self.system, roles=self.roles, messages=[[x, y] for x, y in self.messages], offset=self.offset, sep_style=self.sep_style, sep=self.sep, sep2=self.sep2)

    def dict(self):
        return {
            "system": self.system,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
            "sep": self.sep,
            "sep2": self.sep2,
        }


conv_v1_2 = Conversation(
    system="A chat between a curious human and an artificial intelligence assistant. " "The assistant gives helpful, detailed, and polite answers to the human's questions.",
    roles=("Human", "Assistant"),
    messages=(("Human", "What are the key differences between renewable and non-renewable energy sources?"), ("Assistant", "Renewable energy sources are those that can be replenished naturally.\n")),
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)


conv_vicuna_v1_1 = Conversation(
    system="A chat between a curious user and an artificial intelligence assistant. " "The assistant gives helpful, detailed, and polite answers to the user's questions.",
    roles=("USER", "ASSISTANT"),
    version="v1",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
)

conv_video_chatgpt_v1 = Conversation(
    system="You are Video-ChatGPT, a large vision-language assistant. "
    "You are able to understand the video content that the user provides, and assist the user with a variety of tasks using natural language."
    "Follow the instructions carefully and explain your answers in detail based on the provided video.",
    # system="",
    roles=("USER", "ASSISTANT"),
    version="v1",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
)

default_conversation = conv_v1_2
conv_templates = {
    "default": conv_v1_2,
    "video-chatgpt_v1": conv_video_chatgpt_v1,
    "vicuna_v1_1": conv_vicuna_v1_1,
}

if __name__ == "__main__":
    print(default_conversation.get_prompt())
