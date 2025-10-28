"""Text-to-speech generation logic"""

import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.streamers import BaseStreamer
from threading import Thread

from config import (
    MODEL_NAME, START_OF_HUMAN, END_OF_TEXT, END_OF_HUMAN, END_OF_AI,
    TEMPERATURE, TOP_P, REPETITION_PENALTY, REPETITION_CONTEXT_SIZE, MAX_TOKENS
)


class TokenIDStreamer(BaseStreamer):
    """Custom streamer that yields token IDs"""
    def __init__(self, callback):
        self.callback = callback

    def put(self, value):
        """Called by model.generate() with token IDs"""
        if len(value.shape) > 1:
            token_ids = value[0].tolist()
        else:
            token_ids = value.tolist()

        for token_id in token_ids:
            self.callback(token_id)

    def end(self):
        """Called when generation is complete"""
        pass


class TTSGenerator:
    def __init__(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        if torch.cuda.is_available():
            self.device = 'cuda'
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = 'mps'
        else:
            self.device = 'cpu'

    def prepare_input(self, prompt):
        """Build custom input_ids with special tokens"""
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        start_token = torch.tensor([[START_OF_HUMAN]], dtype=torch.int64)
        end_tokens = torch.tensor([[END_OF_TEXT, END_OF_HUMAN]], dtype=torch.int64)
        modified_input_ids = torch.cat([start_token, input_ids, end_tokens], dim=1)
        modified_input_ids = modified_input_ids.to(self.device)

        attention_mask = torch.ones(1, modified_input_ids.shape[1], dtype=torch.int64)
        attention_mask = attention_mask.to(self.device)

        return modified_input_ids, attention_mask

    def generate(self, prompt, audio_writer, max_tokens=MAX_TOKENS):
        """Generate speech tokens from text prompt"""
        modified_input_ids, attention_mask = self.prepare_input(prompt)

        point_1 = time.time()

        # Stream tokens from LLM
        all_token_ids = []

        def on_token_generated(token_id):
            """Callback for each generated token"""
            all_token_ids.append(token_id)
            # print(f"[LLM] Token {len(all_token_ids)}: {token_id}")
            audio_writer.add_token(token_id)

        streamer = TokenIDStreamer(callback=on_token_generated)

        generation_kwargs = dict(
            input_ids=modified_input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            repetition_penalty=REPETITION_PENALTY,
            num_return_sequences=1,
            eos_token_id=END_OF_AI,
            streamer=streamer,
        )

        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        thread.join()

        point_2 = time.time()

        print(f"\n[MAIN] Generation complete. Total tokens: {len(all_token_ids)}")

        # Decode generated text from token IDs
        generated_text = self.tokenizer.decode(all_token_ids, skip_special_tokens=True)

        return {
            'generated_text': generated_text,
            'all_token_ids': all_token_ids,
            'generation_time': point_2 - point_1,
            'point_1': point_1,
            'point_2': point_2
        }
