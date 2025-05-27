import random
from typing import List


class ContentElementTemplate:

    def __call__(self, content, **kwargs) -> str:
        return content


class IndexContentElementTemplate:

    def __call__(self, content, index, **kwargs) -> str:
        return f"element {index}: {content}"


def element_template_factory(name: str):
    if name == "content":
        return ContentElementTemplate
    elif name == "index-content":
        return IndexContentElementTemplate
    else:
        raise NotImplementedError


class ContextHandler:
    """
    A class to construct input prompts based on given images and texts.

    Attributes:
    ----------
    images : list
        A list of input images path

    texts : list
        A list of input texts to be used alongside images for constructing the prompt.
    """

    def __init__(self, config: dict, images: List[str], texts: List[str], order: List = None):
        self._config = config
        self.image_token = self._config.get("image_token", "<image>")
        self.images = images
        self.texts = texts
        assert len(images) == len(texts)

        # element template
        self._template_name = self._config.get("template", "content")
        self._template = element_template_factory(self._template_name)()

        self._format = self._config.get("format", "default")
        assert self._format in ["default", "shuffle", "custom"], self._format
        if self._format == "custom":
            assert order is not None
            assert len(order) == len(self.images)
            self.images = [self.images[i] for i in order]
            self.texts = [self.texts[i] for i in order]
        if self._format == "shuffle":
            image_text_pair = list(zip(self.images, self.texts))
            image_text_pair = random.sample(image_text_pair, len(image_text_pair))
            self.images, self.texts = zip(*image_text_pair)

    def construct_context(self, start_index: int = 0, end_index: int = -1, index_offset: int = 0):
        # [start_index, end_index)
        selected_images = self.images[start_index : end_index if end_index != -1 else None]
        selected_texts = self.texts[start_index : end_index if end_index != -1 else None]

        image_path = []
        context = []
        for i, (text, image) in enumerate(zip(selected_texts, selected_images)):
            if text == "":
                context.append(self._template(content=self.image_token, index=i + index_offset))
                image_path.append(image)
            else:
                context.append(self._template(content=f'"{text}"', index=i + index_offset))

        if len(context) == 0:
            return "null", image_path
        return ", ".join(context), image_path
