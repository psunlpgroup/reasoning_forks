import time
from typing import Any, List, Optional
import random
from math import ceil
from eval_types import MessageList, ResponseChoice, SamplerBase, SamplerResponse

try:
    from vllm import LLM, SamplingParams
except ImportError:
    LLM = None
    SamplingParams = None

class VLLMSampler(SamplerBase):
    """
    Sampler using vllm engine.
    """
    def __init__(
        self,
        model_name_or_path: str,
        tensor_parallel_size: int = 1,
        temperature: float = 0.5,
        max_tokens: int = 1024,
        top_k: int = 20,
        top_p: float = 0.8,
        tokenizer: Any = None,
        question_prompt_template: Optional[Any] = None,
        trust_remote_code: bool = False,
        **llm_kwargs
    ):
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_k = top_k
        self.top_p = top_p
        self.tokenizer = tokenizer
        self.question_prompt_template = question_prompt_template
        self.model_name_or_path = model_name_or_path

        # Defensive import to allow fail if vllm not available
        if LLM is None:
            raise ImportError(
                "vllm is not installed. Please install vllm to use VLLMSampler."
            )
        # vllm.LLM can be heavy to initialize, so allow multiple kwargs
        print("Tokenizer:", tokenizer)
        self.llm = LLM(
            model=model_name_or_path,
            tokenizer=tokenizer,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=trust_remote_code,
            **llm_kwargs
        )

    def can_apply_chat_template(self):
        # If tokenizer supports chat template
        return (self.tokenizer is not None) and (hasattr(self.tokenizer, "apply_chat_template"))

    def apply_chat_template(self, messages):
        assert self.can_apply_chat_template()
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return prompt

    def convert_message_list_to_prompt(self, message_list: MessageList) -> str:
        # By default, just join as plain text, or use question_prompt_template if available
        if self.can_apply_chat_template():
            return self.apply_chat_template(message_list)
        elif self.question_prompt_template is not None:
            return self.question_prompt_template(message_list)
        else:
            # Fallback: naive concatenation
            return "\n".join([f"{msg['role']}: {msg['content']}" for msg in message_list])

    def complete(self, prompts: str, n: int = 1, stop=None) -> SamplerResponse:
        sampling_params = SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_k=self.top_k,
            top_p=self.top_p,
            n=n,
            include_stop_str_in_output=True,
            stop=stop,
        )
        resonpses = []
        try:
            # vllm supports batch generation, each prompt can be replicated n times
            outputs = self.llm.generate(prompts, sampling_params)
            for out in outputs:
                choices = []
                completion_tokens = []
                # Each output may contain multiple generations
                for g in out.outputs:
                    response_text = g.text
                    choices.append(ResponseChoice(response_text=response_text))
                    # Record the number of generated tokens for this choice
                    completion_tokens.append(len(g.token_ids))
                
                # Retrieve prompt tokens for thoroughness 
                prompt_tokens = len(out.prompt_token_ids) if hasattr(out, "prompt_token_ids") else 0

                resonpses.append(
                    SamplerResponse(
                        status="OK",
                        choices=choices,
                        response_metadata={
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens # List of token counts matching choices
                        },
                        actual_queried_message_list=[{"prompt": out.prompt}],
                    )
                )
            return resonpses
        except Exception as e:
            print("VLLM Exception during sampling:", e)
            return [SamplerResponse(
                    status=f"Error: {e!r}",
                    choices=[],
                    response_metadata={},
                    actual_queried_message_list=[{"prompt": ""}],
                )]

    def __call__(self, message_list: MessageList, n: int = 1) -> SamplerResponse:
        # Apply chat template to convert to prompt as needed
        prompt = self.convert_message_list_to_prompt(message_list)
        return self.complete(prompt, n=n)


class DiverPathVLLMSampler(VLLMSampler):
    def __init__(
        self,
        model_name_or_path: str,
        first_topk: int,
        tensor_parallel_size: int = 1,
        temperature: float = 0.5,
        max_tokens: int = 1024,
        top_k: int = 20,
        top_p: float = 0.8,
        tokenizer: Any = None,
        question_prompt_template: Optional[Any] = None,
        trust_remote_code: bool = False,
        **kwargs,
    ):
        self.first_topk = first_topk
        print("first_topk:", self.first_topk)
        super().__init__(
            model_name_or_path=model_name_or_path,
            tensor_parallel_size=tensor_parallel_size,
            temperature=temperature,
            max_tokens=max_tokens,
            top_k=top_k,
            top_p=top_p,
            tokenizer=tokenizer,
            question_prompt_template=question_prompt_template,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )


    def complete(self, prompts, n=1, stop_strs=None) -> SamplerResponse:
        resonpses = []
        try:
            firsttoken_sampling_params = SamplingParams(
                temperature=self.temperature,
                max_tokens=1,
                top_p=1.0,
                top_k=self.first_topk,
                logprobs=self.first_topk,
                n=1,
            )
            firsttoken_outputs = self.llm.generate(prompts, firsttoken_sampling_params)

            # Gather top-k candidates for the first position for each prompt
            firsttoken_candidates = []
            for idx, out in enumerate(firsttoken_outputs):
                candidates = []
                if hasattr(out, "outputs") and out.outputs:
                    candidates = [logprobs_info.decoded_token for tok_id, logprobs_info in out.outputs[0].logprobs[0].items()]
                print(f"[DiverPathVLLMSampler] Prompt {idx} top candidates:", candidates)
                num_repeats = (n + len(candidates) - 1) // len(candidates)
                candidates = (candidates * num_repeats)[:n]
                print(f"[DiverPathVLLMSampler] Prompt {idx} repeated candidates for n={n}:", candidates)
                random.shuffle(candidates)
                firsttoken_candidates.append(candidates)
            
            prefixed_prompts = []
            for idx, (prompt, candidates) in enumerate(zip(prompts, firsttoken_candidates)):
                for c in candidates:
                    concatenated_prompt = prompt + c
                    prefixed_prompts.append(concatenated_prompt)
            sampling_params = SamplingParams(
                temperature=self.temperature,
                max_tokens=self.max_tokens - 1,
                top_k=self.top_k,
                top_p=self.top_p,
                n=1,
            )
            print("[DiverPathVLLMSampler] Number of prefixed prompts:", len(prefixed_prompts))
            outputs = self.llm.generate(prefixed_prompts, sampling_params)
            
            num_chunks = ceil(len(outputs) / n)
            for i in range(num_chunks):
                chunk = outputs[i * n: (i + 1) * n]
                choices = []
                completion_tokens = []
                
                for out, first_tok in zip(chunk, firsttoken_candidates[i]):
                    composed_response = first_tok + out.outputs[0].text
                    choices.append(ResponseChoice(response_text=composed_response))
                    
                    # 1 token for the forced prefix + the rest of the generated tokens
                    token_count = 1 + len(out.outputs[0].token_ids)
                    completion_tokens.append(token_count)
                    
                assert len(choices) == n
                prompts_for_chunk = [{"prompt": chunk[0].prompt}]
                
                # For the base prompt tokens, subtract 1 to get the original prompt length before the prefix was added
                prompt_tokens = (len(chunk[0].prompt_token_ids) - 1) if hasattr(chunk[0], "prompt_token_ids") else 0

                resonpses.append(
                    SamplerResponse(
                        status="OK",
                        choices=choices,
                        response_metadata={
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens # List of token counts matching choices
                        },
                        actual_queried_message_list=prompts_for_chunk,
                    )
                )
            return resonpses
        except Exception as e:
            print("VLLM Exception during sampling:", e)
            return [SamplerResponse(
                    status=f"Error: {e!r}",
                    choices=[],
                    response_metadata={},
                    actual_queried_message_list=[{"prompt": ""}],
                )]