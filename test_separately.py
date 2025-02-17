import torch
from transformers import AutoModelForCausalLM
from pathlib import Path

from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
from deepseek_vl2.utils.io import load_pil_images_from_paths


# specify the path to the model
model_path = "deepseek-ai/deepseek-vl2-tiny"
vl_chat_processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer

vl_gpt: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
vl_gpt = vl_gpt.to(torch.bfloat16).cpu().eval()

images_paths = ["/home/albert/Documents/research_dump_bin/DeepSeek-VL2/images/image.png"]
pil_images = load_pil_images_from_paths(images_paths)


tokenized_image, images_list, images_seq_mask, images_spatial_crop, num_image_tokens, best_width, best_height = vl_chat_processor.encode_images(pil_images)

with torch.inference_mode():
    projected_features = vl_gpt.encode_images(images_list[None, ...], images_spatial_crop[None, ...])

conversation = [
    {
        "role": "<|User|>",
        "content": "This is image_1: <image>\n What is the relationship between the plant (spatially located at [-13.76, 21.82, 1.97] meters) and the couch (spatially located at [-12.48, 22.37, 1.53])?",
    },
    {"role": "<|Assistant|>", "content": ""}
]
prepare_inputs = vl_chat_processor.encode_conversations_and_features(conversation,
                                                                     best_width,
                                                                     best_height,
                                                                     projected_features,
                                                                     images_spatial_crop,
                                                                     num_image_tokens)

# run image encoder to get the image embeddings
inputs_embeds = vl_gpt.prepare_input_embeds_from_feats(**prepare_inputs)

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

answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
print("Response:")
print(answer)
# print(f"{prepare_inputs['sft_format'][0]}", answer)
