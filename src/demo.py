"""
小红书内容审核助手 Demo
基于 Qwen2-VL + LoRA SFT 微调
"""

import gradio as gr
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

# ============ 配置 ============
MODEL_PATH = "D:/2025 Content Review Assistant/LLaMA-Factory/models/qwen2vl-content-review-sft-merged" # 放置模型的路径

SYSTEM_PROMPT = """你是小红书内容审核助手，负责判断用户发布的内容是否符合平台规范。

请根据以下规则进行审核：
1. 禁止虚假宣传和夸大功效
2. 禁止引流到私域（微信、QQ等）
3. 禁止违规医疗健康声明
4. 禁止低俗、暴力、违法内容
5. 禁止抄袭和侵权内容
6. 禁止诱导互动（求赞、求关注）
7. 禁止欺诈和非法服务

审核结果分为：通过(pass)、需要修改(needs_edit)、移除(remove)、升级人工审核(escalate)"""

# ============ 加载模型 ============
print("正在加载模型...")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(MODEL_PATH)
print("模型加载完成！")


def review_content(text: str, image=None) -> str:
    """审核内容"""
    if not text.strip():
        return "请输入待审核的文案内容"
    
    # 构建消息
    user_content = f"请审核这段文案：「{text}」"
    
    if image is not None:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": user_content}
            ]}
        ]
    else:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ]
    
    # 处理输入
    text_input = processor.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    if image is not None:
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text_input],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(model.device)
    else:
        inputs = processor(
            text=[text_input],
            padding=True,
            return_tensors="pt"
        ).to(model.device)
    
    # 生成
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            pad_token_id=processor.tokenizer.pad_token_id
        )
    
    # 解码
    generated_ids = outputs[:, inputs.input_ids.shape[1]:]
    response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    return response


# ============ Gradio 界面 ============
with gr.Blocks(
    title="小红书内容审核助手",
    theme=gr.themes.Soft()
) as demo:
    
    gr.Markdown("""
    # 🔍 小红书内容审核助手
    
    基于 **Qwen2-VL-2B + LoRA SFT** 微调的多模态内容审核系统
    
    **审核类别：** 通过 ✅ | 需要修改 ✏️ | 违规删除 ❌ | 人工复核 👤
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            text_input = gr.Textbox(
                label="📝 待审核文案",
                placeholder="输入小红书风格的文案，例如：分享今天的穿搭look～",
                lines=4
            )
            image_input = gr.Image(
                label="🖼️ 配图（可选）", 
                type="pil"
            )
            submit_btn = gr.Button("🚀 开始审核", variant="primary", size="lg")
        
        with gr.Column(scale=1):
            output = gr.Textbox(
                label="📋 审核结果",
                lines=12,
                show_copy_button=True
            )
    
    # 示例
    gr.Markdown("### 💡 测试样例")
    gr.Examples(
        examples=[
            ["分享今天做的午餐，番茄炒蛋，简单又好吃～", None],
            ["这款美白霜用了一周，皮肤白了三个色号！效果太惊艳了", None],
            ["私我领取内部优惠券，比官方便宜50%！仅限前100名", None],
            ["姐妹们这个减肥药真的有用！一个月瘦了20斤不反弹", None],
            ["回购了无数次的面膜，用习惯了离不开", None],
            ["点赞过1000就抽奖，帮帮忙", None],
        ],
        inputs=[text_input, image_input],
        label="点击下方示例快速测试"
    )
    
    submit_btn.click(
        fn=review_content, 
        inputs=[text_input, image_input], 
        outputs=output
    )
    
    gr.Markdown("""
    ---
    **项目信息：** 基于 LLaMA-Factory 框架 | 训练数据 1000+ 条 | RTX 4060 训练约 12 分钟
    """)


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True  # 生成公网链接
    )
