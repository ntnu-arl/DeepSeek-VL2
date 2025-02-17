import torch
from transformers import AutoModelForCausalLM

from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
from deepseek_vl2.utils.io import load_pil_images


# specify the path to the model
model_path = "deepseek-ai/deepseek-vl2-tiny"
vl_chat_processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer

vl_gpt: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
vl_gpt = vl_gpt.to(torch.bfloat16).cpu().eval()

## single image conversation example
## Please note that <|ref|> and <|/ref|> are designed specifically for the object localization feature. These special tokens are not required for normal conversations.
## If you would like to experience the grounded captioning functionality (responses that include both object localization and reasoning), you need to add the special token <|grounding|> at the beginning of the prompt. Examples could be found in Figure 9 of our paper.
conversation = [
    {
        "role": "<|User|>",
        "content": "This is image_1: <image>\n What is the relationship between the plant (spatially located at [-13.76, 21.82, 1.97] meters) and the couch (spatially located at [-12.48, 22.37, 1.53])?",
        "images": [
            "/home/albert/Documents/research_dump_bin/DeepSeek-VL2/images/image.png",
        ],
    },
    {"role": "<|Assistant|>", "content": ""}
]

# conversation = [
#     {
#         "role": "<|User|>",
#         "content": "This is image_1: <image>\n"
#                    "This is image_2: <image>\n"
#                    "This is image_3: <image>\n Can you tell me what are in the images?",
#         "images": [
#             "./DeepSeek-VL2/images/multi_image_1.jpeg",
#             "./DeepSeek-VL2/images/multi_image_2.jpeg",
#             "./DeepSeek-VL2/images/multi_image_3.jpeg",
#         ],
#     },
#     {"role": "<|Assistant|>", "content": ""}
# ]

# load images and prepare for inputs
pil_images = load_pil_images(conversation)
prepare_inputs = vl_chat_processor(
    conversations=conversation,
    images=pil_images,
    force_batchify=True,
    system_prompt=""
).to(vl_gpt.device)

# run image encoder to get the image embeddings
inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

# run the model to get the response
outputs = vl_gpt.language.generate(
    inputs_embeds=inputs_embeds,
    attention_mask=prepare_inputs.attention_mask,
    pad_token_id=tokenizer.eos_token_id,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    max_new_tokens=512,
    do_sample=False,
    use_cache=True
)

answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=False)
print(f"{prepare_inputs['sft_format'][0]}", answer)