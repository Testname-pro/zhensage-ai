"""
下载 DialoGPT-small 模型到本地
使用国内镜像源
"""

import os

# 设置国内镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "microsoft/DialoGPT-small"
SAVE_DIR = "/workspace/dialogpt_model"

print(f"正在从 hf-mirror.com 下载模型: {MODEL_NAME}")
print(f"保存到: {SAVE_DIR}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

tokenizer.save_pretrained(SAVE_DIR)
model.save_pretrained(SAVE_DIR)

print(f"模型下载完成! 已保存到 {SAVE_DIR}")
print(f"模型大小: {sum(os.path.getsize(os.path.join(SAVE_DIR, f)) for f in os.listdir(SAVE_DIR)) / 1024 / 1024:.1f} MB")
