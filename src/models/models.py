import gc
from abc import ABC, abstractmethod
from multiprocessing import Process, Queue
from typing import Any, Dict, List, Mapping, Optional, Tuple

import torch
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from openai import OpenAI
from qwen_vl_utils import process_vision_info
from transformers import (
    # Qwen2_5_VLForConditionalGeneration,
    AutoModelForCausalLM,
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
)

import src.utils.images as images_utils

#######
# INTERFACES
#######


class ModelInterface(LLM, ABC):
    """
    Abstract base class for model interfaces.
    """

    model_name: Optional[str] = None
    capabilities: List[str] = []
    skip_special_tokens: bool = False
    max_token: int = 10000
    temperature: float = 0.01
    top_p: float = 0.9
    history_len: int = 0
    openai_server: Optional[str] = None
    api_key: Optional[str] = None

    @property
    def _llm_type(self) -> str:
        return self.__class__.__name__

    @property
    def _history_len(self) -> int:
        return self.history_len

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            "max_token": self.max_token,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "history_len": self.history_len,
        }

    @abstractmethod
    def _call(
        self,
        prompt: str,
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """
        Run the LLM on the given input.

        Performs an inference using qwen-vl based models.

        Args:
            prompt: The prompt to generate from.
            stop: Stop words to use when generating. Model output is cut off at the
                first occurrence of any of the stop substrings.
                If stop tokens are not supported consider raising NotImplementedError.
            run_manager: Callback manager for the run.
            **kwargs: Arbitrary additional keyword arguments. These are usually passed
                to the model provider API call.

        Returns:
            str: The model output as a string. SHOULD NOT include the prompt.
        """
        pass

    @abstractmethod
    def load_model(self) -> Tuple[Any, ...]:
        """
        Loads and returns the model to make inferences on.

        Returns:
            Tuple[Any, ...]: The loaded model and any additional components.
        """
        pass

    def unload(self, *args: Any) -> None:
        """
        Unloads the given items and extras from cuda memory.

        Args:
            *args (Any): The items to unload.
        """
        try:
            for arg in args:
                del arg
            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            pass

    def __call__(
        self, prompt, stop=None, callbacks=None, *, tags=None, metadata=None, **kwargs
    ) -> str:
        """
        Calls the model with the given prompt and additional arguments.

        Args:
            prompt (str): The prompt to generate from.
            stop (Optional[List[str]]): Stop words to use when generating.
            callbacks (Optional[Any]): Callbacks for the run.
            tags (Optional[Any]): Tags for the run.
            metadata (Optional[Any]): Metadata for the run.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            str: The model output as a string.
        """
        return self._call(
            prompt=prompt,
            stop=stop,
            callbacks=callbacks,
            tags=tags,
            metadata=metadata,
            **kwargs,
        )


#######
# IMPLEMENTATION
#######


class TextModel(ModelInterface):
    """
    A model interface for text-only inference.
    """

    capabilities: List[str] = ["text"]

    def __init__(self, model_name: str, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.model_name: str = model_name

    def inference(
        self,
        sys_prompt: str,
        user_prompt: str,
        max_tokens: int = 512,
        **kwargs: Any,
    ) -> None:
        """
        Performs an inference using a text-only model.
        """
        assert isinstance(sys_prompt, str), "sys_prompt must be a string"
        assert isinstance(user_prompt, str), "user_prompt must be a string"
        assert isinstance(max_tokens, int), "max_tokens must be an integer"

        messages: List[Dict[str, Any]] = []
        processed_output_text: str
        if self.openai_server:
            if sys_prompt:
                messages.append({"role": "system", "content": sys_prompt})
            messages.append({"role": "user", "content": user_prompt})
            client = OpenAI(base_url=self.openai_server, api_key=self.api_key)
            processed_output_text = (
                client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,  # type: ignore
                    max_tokens=max_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                )
                .choices[0]
                .message.content
            )
            kwargs["result_queue"].put(processed_output_text)
        else:
            if sys_prompt:
                messages.append({"role": "system", "content": sys_prompt})
            messages.append({"role": "user", "content": user_prompt})
            processed_output_text = self._local_inference(
                messages, max_tokens=max_tokens, **kwargs
            )
            kwargs["result_queue"].put(processed_output_text)

    def _local_inference(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int = 512,
        **kwargs: Any,
    ) -> str:
        model, processor = self.load_model()

        text: str = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = processor(
            text=[text],
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Inference: Generate output tokens
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=max_tokens)

        generated_ids_trimmed: List[torch.Tensor] = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        processed_output_text: str = next(
            iter(
                processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=self.skip_special_tokens,
                    clean_up_tokenization_spaces=False,
                )
            ),
            "",
        )

        self.unload(model, processor, inputs, generated_ids, generated_ids_trimmed)
        return processed_output_text

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """
        Wraps inference in a separate process to help free GPU memory.
        """
        assert isinstance(prompt, str), "prompt must be a string"
        assert stop is None or isinstance(stop, list), "stop must be None or a list"

        result_queue: Queue = Queue()
        max_tokens: int = kwargs.pop("max_tokens", 512)
        sys_prompt: str = kwargs.pop("sys_prompt", "")

        kwargs["result_queue"] = result_queue
        p = Process(
            target=self.inference,
            args=(sys_prompt, prompt, max_tokens),
            kwargs=kwargs,
        )
        p.start()
        p.join()

        result = result_queue.get()
        assert isinstance(result, str), "result must be a string"
        return result

    def load_model(self) -> Tuple[Any, AutoProcessor]:
        """
        Loads and returns the text-only model and processor.
        """
        assert isinstance(self.model_name, str), "model_name must be a string"
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype="auto", device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(self.model_name)
        return model, processor


class VisionModel(ModelInterface):
    """
    A model interface for vision+text inference.
    """

    capabilities: List[str] = ["image", "text"]

    def __init__(self, model_name: str, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.model_name: str = model_name

    def inference(
        self,
        sys_prompt: str,
        user_prompt: str,
        max_tokens: int = 512,
        **kwargs: Any,
    ) -> None:
        """
        Performs an inference using a vision+text model.
        """
        assert isinstance(sys_prompt, str), "sys_prompt must be a string"
        assert isinstance(user_prompt, str), "user_prompt must be a string"
        assert isinstance(max_tokens, int), "max_tokens must be an integer"

        messages: List[Dict[str, Any]] = []
        processed_output_text: str
        if self.openai_server:
            if sys_prompt:
                messages.append({"role": "system", "content": sys_prompt})
            messages.append(
                {
                    "role": "user",
                    "content": [
                        *[
                            {
                                "type": t,
                                t: val,
                            }
                            if t != "image"
                            else {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{images_utils.im_2_b64(val)}"
                                },
                            }
                            for t, val in kwargs.items()
                            if t in self.capabilities
                        ],
                        {"type": "text", "text": user_prompt},
                    ],
                }
            )
            client = OpenAI(base_url=self.openai_server, api_key=self.api_key)
            processed_output_text = (
                client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,  # type: ignore
                    max_tokens=max_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                )
                .choices[0]
                .message.content
            )
            kwargs["result_queue"].put(processed_output_text)
        else:
            if sys_prompt:
                messages.append({"role": "system", "content": sys_prompt})
            messages.append(
                {
                    "role": "user",
                    "content": [
                        *[
                            {
                                "type": t,
                                t: val,
                            }
                            for t, val in kwargs.items()
                            if t in self.capabilities
                        ],
                        {"type": "text", "text": user_prompt},
                    ],
                }
            )
            processed_output_text = self._local_inference(
                messages, max_tokens=max_tokens, **kwargs
            )
            kwargs["result_queue"].put(processed_output_text)

    def _local_inference(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int = 512,
        **kwargs: Any,
    ) -> str:
        model, processor = self.load_model()

        text: str = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=max_tokens)

        generated_ids_trimmed: List[torch.Tensor] = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        processed_output_text: str = next(
            iter(
                processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=self.skip_special_tokens,
                    clean_up_tokenization_spaces=False,
                )
            ),
            "",
        )

        self.unload(model, processor, inputs, generated_ids, generated_ids_trimmed)
        return processed_output_text

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """
        Wraps vision+text inference in a separate process.
        """
        assert isinstance(prompt, str), "prompt must be a string"
        assert stop is None or isinstance(stop, list), "stop must be None or a list"

        result_queue: Queue = Queue()
        max_tokens: int = kwargs.pop("max_tokens", 512)
        sys_prompt: str = kwargs.pop("sys_prompt", "")

        kwargs["result_queue"] = result_queue
        p = Process(
            target=self.inference,
            args=(sys_prompt, prompt, max_tokens),
            kwargs=kwargs,
        )
        p.start()
        p.join()

        result = result_queue.get()
        assert isinstance(result, str), "result must be a string"
        return result

    def load_model(self) -> Tuple[Any, AutoProcessor]:
        """
        Loads and returns the vision+text model and processor.
        """
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_name, torch_dtype="auto", device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(self.model_name)
        return model, processor


class QwenVLModel(VisionModel):
    """
    Implementation of the QwenVL model interface.
    """

    capabilities: List[str] = ["image", "text"]

    def __init__(
        self,
        model_name: str,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name, *args, **kwargs)

    def load_model(self) -> Tuple[Qwen2VLForConditionalGeneration, AutoProcessor]:
        """
        Loads and returns the model to make inferences on.

        Returns:
            Tuple[Qwen2VLForConditionalGeneration, AutoProcessor]: The loaded model and processor.
        """
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_name, torch_dtype="auto", device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(self.model_name)
        return model, processor


if __name__ == "__main__":
    model = TextModel("Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4")
    sys_prompt = "You will receive a structured event log with the following columns:"
    user_prompt = "1. **ScreenID**: Identifies the group of events corresponding to a specific screen or workflow step."
    print(model(user_prompt, sys_prompt=sys_prompt))
