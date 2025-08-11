import torch
import torch.nn as nn
import math
from deepseek_vl2.models import DeepseekVLV2ForCausalLM, DeepseekVLV2Processor, select_best_resolution
from config import DeepSeekVL2Config


class DeepSeekVL2(nn.Module):
    def __init__(self, config: DeepSeekVL2Config, verbose=False):
        super().__init__()
        self.config = config
        self.verbose = verbose
        self.model = DeepseekVLV2ForCausalLM.from_pretrained(
            config.model_name, trust_remote_code=True
        ).to(torch.bfloat16).eval()
        self.processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(
            config.model_name
        )
        self.tokenizer = self.processor.tokenizer
        self._canary_param = nn.Parameter(torch.empty(0))

        if self.config.cropping:
            self.best_width, self.best_height = select_best_resolution(
                (self.config.height, self.config.width),
                self.processor.candidate_resolutions,
            )
        else:
            self.best_width, self.best_height = (
                self.processor.image_size,
                self.processor.image_size,
            )
        num_width_tiles, num_height_tiles = (
            self.best_width // self.processor.image_size,
            self.best_height // self.processor.image_size,
        )
        h = w = math.ceil(
            (self.processor.image_size // self.processor.patch_size)
            / self.processor.downsample_ratio
        )
        self.images_spatial_crop = torch.tensor(
            [[num_width_tiles, num_height_tiles]], dtype=torch.long
        )
        self.num_image_tokens = [
            int(h * (w + 1) + 1 + (num_height_tiles * h) * (num_width_tiles * w + 1))
        ]
        self.feature_channels = 1 + math.ceil(
            self.best_height / self.processor.image_size
        ) * math.ceil(self.best_width / self.processor.image_size)

    @property
    def device(self):
        return self._canary_param.device

    def generate_caption(self, image_embeds, prompts, deterministic=True):
        """Generate image caption."""
        image_embeds = image_embeds.to(torch.bfloat16)
        answers = []
        for image_embed, prompt in zip(image_embeds, prompts):
            conversation = [
                {
                    "role": "<|User|>",
                    "content": f"<image>\n {prompt}",
                },
                {"role": "<|Assistant|>", "content": ""},
            ]

            prepare_inputs = self.processor.encode_conversations_and_features(
                conversation,
                self.best_width,
                self.best_height,
                image_embed.reshape(
                    self.feature_channels,
                    image_embed.shape[0] // self.feature_channels,
                    image_embed.shape[1],
                ),
                self.images_spatial_crop,
                self.num_image_tokens,
            ).to(self.device)
            with torch.no_grad():
                inputs_embeds = self.model.prepare_input_embeds_from_feats(**prepare_inputs)
                if deterministic:
                    outputs = self.model.language.generate(
                        inputs_embeds=inputs_embeds,
                        attention_mask=prepare_inputs.attention_mask,
                        pad_token_id=self.tokenizer.eos_token_id,
                        bos_token_id=self.tokenizer.bos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        max_new_tokens=512,
                        do_sample=False,
                        use_cache=True,
                    )
                else:
                    outputs = self.model.generate(
                        inputs_embeds=inputs_embeds,
                        input_ids=prepare_inputs.input_ids,
                        images=prepare_inputs.images,
                        images_seq_mask=prepare_inputs.images_seq_mask,
                        images_spatial_crop=prepare_inputs.images_spatial_crop,
                        attention_mask=prepare_inputs.attention_mask,
                        past_key_values=None,
                        pad_token_id=self.tokenizer.eos_token_id,
                        bos_token_id=self.tokenizer.bos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        max_new_tokens=512,
                        do_sample=True,
                        temperature=0.4,
                        top_p=0.9,
                        repetition_penalty=1.1,
                        use_cache=True,
                    )
                answer = self.tokenizer.decode(
                            outputs[0].cpu().tolist(), skip_special_tokens=True
                         )
                answer = answer.replace(f": \n {prompt}\n\n:", "")
                answers.append(answer.strip())
                

        return answers
