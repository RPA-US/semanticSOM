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
    AutoModel,
    AutoProcessor,
    AutoTokenizer,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    BitsAndBytesConfig,
)

from PIL import Image
from src.models.utils import load_image

from src.semantics.prompts import COT_ACTION_TARGET_BASE

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

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._loaded: bool = False
        self._loaded_model: Optional[Any] = None
        self._loaded_processor: Optional[Any] = None

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

    @property
    def loaded(self) -> bool:
        return self._loaded

    def manual_load(self) -> None:
        if not self._loaded:
            self._loaded_model, self._loaded_processor = self.load_model()
            self._loaded = True

    def manual_unload(self) -> None:
        if self._loaded:
            self.unload(self._loaded_model, self._loaded_processor)
            self._loaded = False

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

    **Example Usage (Local Inference):**

    ```python
    from models.models import TextModel

    # Using a locally loaded model
    model = TextModel("path/to/local/text-model")
    output = model("Hello world!", sys_prompt="Provide a short greeting")
    print(output)
    ```

    **Example Usage (Using OpenAI API):**

    ```python
    from models.models import TextModel

    # Configure for OpenAI API inference
    model = TextModel("your-model-id")
    model.openai_server = "https://api.openai.com/v1"
    model.api_key = "YOUR_API_KEY"
    output = model("Hello world!", sys_prompt="Provide a short greeting")
    print(output)
    ```
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
    ) -> Optional[str]:
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
        else:
            if sys_prompt:
                messages.append({"role": "system", "content": sys_prompt})
            messages.append({"role": "user", "content": user_prompt})
            processed_output_text = self._local_inference(
                messages, max_tokens=max_tokens, **kwargs
            )
        if "result_queue" in kwargs:
            kwargs["result_queue"].put(processed_output_text)
            return None
        else:
            return processed_output_text

    def _local_inference(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int = 512,
        **kwargs: Any,
    ) -> str:
        if self.loaded:
            model: AutoModelForCausalLM
            processor: AutoProcessor
            model, processor = self._loaded_model, self._loaded_processor
        else:
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

        if self.loaded:
            del model
            del processor
        del inputs
        del generated_ids
        del generated_ids_trimmed

        torch.cuda.empty_cache()

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
        max_tokens: int = kwargs.pop("max_tokens", 512)
        sys_prompt: str = kwargs.pop("sys_prompt", "")

        if self.loaded:
            result = self.inference(sys_prompt, prompt, max_tokens, **kwargs)
        else:
            result_queue: Queue = Queue()

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

    def load_model(self) -> Tuple[AutoModelForCausalLM, AutoProcessor]:
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

    **Example Usage (Local Inference):**

    ```python
    from models.models import VisionModel

    # Using a locally loaded model
    model = TextModel("hugginface/identifier")
    output = model("Hello world!", sys_prompt="Provide a short greeting")
    print(output)
    ```

    **Example Usage (Using OpenAI API):**

    ```python
    from models.models import VisionModel

    # Configure for OpenAI API inference
    model = TextModel("your-model-id")
    model.openai_server = "https://api.openai.com/v1"
    model.api_key = "YOUR_API_KEY"
    output = model("Hello world!", sys_prompt="Provide a short greeting")
    print(output)
    ```
    """

    capabilities: List[str] = ["image", "text"]

    def __init__(self, model_name: str, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.model_name: str = model_name

    def inference(
        self,
        sys_prompt: str,
        user_prompt: str,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> Optional[str]:
        """
        Performs an inference using a vision+text model.
        """
        assert isinstance(sys_prompt, str), "sys_prompt must be a string"
        assert isinstance(user_prompt, str), "user_prompt must be a string"
        assert isinstance(max_tokens, int), "max_tokens must be an integer"

        messages: List[Dict[str, Any]] = []
        processed_output_text: str
        if self.openai_server:
            # if sys_prompt:
            #     # messages.append({"role": "system", "content": sys_prompt}) # Not all models support system messages
            #     messages.append({"role": "user", "content": sys_prompt})
            sys_prompt = sys_prompt if sys_prompt else "You are a helpful assistant"
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"{sys_prompt}<image>"},
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
                                t: val
                                if t != "image"
                                else images_utils.resize_img(val),
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
        if "result_queue" in kwargs:
            kwargs["result_queue"].put(processed_output_text)
            return None
        else:
            return processed_output_text

    def _local_inference(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int = 512,
        **kwargs: Any,
    ) -> str:
        if self.loaded:
            model: AutoModel
            processor: AutoProcessor
            model, processor = self._loaded_model, self._loaded_processor
        else:
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
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True,
                pad_token_id=151645,
            )

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
        max_tokens: int = kwargs.pop("max_tokens", 512)
        sys_prompt: str = kwargs.pop("sys_prompt", "")

        if self.loaded:
            result = self.inference(sys_prompt, prompt, max_tokens, **kwargs)
        else:
            result_queue: Queue = Queue()

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

    def load_model(self) -> Tuple[AutoModel, AutoProcessor]:
        """
        Loads and returns the vision+text model and processor.
        """
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_name, torch_dtype="auto", device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(self.model_name)
        return model, processor


class InternVLModel(VisionModel):
    def inference(
        self,
        sys_prompt: str,
        user_prompt: str,
        max_tokens: int = 512,
        **kwargs: Any,
    ) -> Optional[str]:
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
        else:
            question = ""
            if sys_prompt:
                question += sys_prompt + "\n"
            question += user_prompt
            generation_config = dict(
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
                pad_token_id=151645,
            )
            pixel_values = None
            if "image" in kwargs:
                question = f"<image>\n{question}"

                # Wide images are badly processed. Thus, we are going to make sure the image has an aspect ratio of 16:9, with a resolution of 1920 x 1080
                # But to not stretch the image, we will do it by adding black padding and putting the original image at the center
                image = kwargs["image"]
                width, height = image.size
                aspect_ratio = width / height
                if aspect_ratio < 16 / 9:
                    new_width = int(height * 16 / 9)
                    padding = (new_width - width) // 2
                    image = images_utils.add_padding(image, padding, 0)
                elif aspect_ratio > 16 / 9:
                    new_height = int(width * 9 / 16)
                    padding = (new_height - height) // 2
                    image = images_utils.add_padding(image, 0, padding)
                image = image.resize((1920, 1080))

                pixel_values = (
                    load_image(image, max_num=12)
                    .to(torch.bfloat16 if "Intern" in self.model_name else torch.int8)
                    .cuda()
                )
            processed_output_text = self._local_inference(
                question, generation_config, pixel_values
            )
        if "result_queue" in kwargs:
            kwargs["result_queue"].put(processed_output_text)
            return None
        else:
            return processed_output_text

    def _local_inference(  # type: ignore[override]
        self,
        question: str,
        generation_config: Dict[str, Any],
        pixel_values=None,
    ) -> str:
        if self.loaded:
            model: AutoModel
            tokenizer: AutoTokenizer
            model, tokenizer = self._loaded_model, self._loaded_processor
        else:
            model, tokenizer = self.load_model()

        response = model.chat(
            tokenizer,
            pixel_values,
            question,
            generation_config,
            history=None,
            return_history=False,
        )

        self.unload(model, tokenizer, pixel_values, question)
        return response

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
        max_tokens: int = kwargs.pop("max_tokens", 1024)
        sys_prompt: str = kwargs.pop("sys_prompt", "")

        if self.loaded:
            result = self.inference(sys_prompt, prompt, max_tokens, **kwargs)
        else:
            result_queue: Queue = Queue()

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

    def load_model(self) -> Tuple[AutoModel, AutoTokenizer]:
        """
        Loads and returns the vision+text model and processor.
        """
        model = (
            AutoModel.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                use_flash_attn=True,
                trust_remote_code=True,
            )
            .eval()
            .cuda()
        )
        model.generation_config.pad_token_id = model.generation_config.eos_token_id
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True, use_fast=False
        )
        return model, tokenizer


class NVLModel(InternVLModel):
    def load_model(self) -> Tuple[AutoModel, AutoTokenizer]:
        """
        Loads and returns the vision+text model and processor.
        """
        model = AutoModel.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            quantization_config=BitsAndBytesConfig(
                load_in_8bit=True,
            ),
            low_cpu_mem_usage=True,
            use_flash_attn=False,
            device_map="cuda",
            trust_remote_code=True,
        ).eval()
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True, use_fast=False
        )
        return model, tokenizer


class QwenVLModel(VisionModel):
    """
    Implementation of the QwenVL model interface.

    **Example Usage (Local Inference):**

    ```python
    from models.models import QwenVLModel

    # Using a locally loaded QwenVL model
    model = QwenVLModel("Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4")
    image = Image.open("path/to/image.jpg")
    output = model(
        "Provide insights on the image", sys_prompt="Image analysis", image=image
    )
    print(output)
    ```

    **Example Usage (Using OpenAI API):**

    ```python
    from models.models import QwenVLModel

    # Configure for OpenAI API inference
    model = QwenVLModel("your-qwenvl-model-id")
    model.openai_server = "https://api.openai.com/v1"
    model.api_key = "YOUR_API_KEY"
    image = Image.open("path/to/image.jpg")
    output = model(
        "Provide insights on the image", sys_prompt="Image analysis", image=image
    )
    print(output)
    ```
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
            self.model_name,
            torch_dtype="auto",
            device_map="cuda",
            quantization_config=BitsAndBytesConfig(
                load_in_8bit=True,
            )
            if "QVQ" in self.model_name
            else None,
        )
        processor = AutoProcessor.from_pretrained(self.model_name)
        return model, processor


class Qwen2_5VLModel(QwenVLModel):
    def load_model(self) -> Tuple[Qwen2_5_VLForConditionalGeneration, AutoProcessor]:
        """
        Loads and returns the model to make inferences on.

        Returns:
            Tuple[Qwen2VLForConditionalGeneration, AutoProcessor]: The loaded model and processor.
        """
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if "AWQ" in self.model_name else "auto",
            device_map="auto",
        )
        processor = AutoProcessor.from_pretrained(self.model_name)
        return model, processor


if __name__ == "__main__":
    model = VisionModel(model_name="NVlabs/NVILA-15B", openai_server="localhost:8000")
    sys_prompt = COT_ACTION_TARGET_BASE
    # print(model("1+1", sys_prompt=sys_prompt))
    # print(model("1+2", sys_prompt=sys_prompt))
    image = Image.open("temp.png")
    print(
        model(
            "Describe the object with a red highlighted sqare around it",
            sys_prompt=sys_prompt,
            image=image,
        )
    )
