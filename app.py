"""
帧神AI (Zhensage AI) — Flask 后端
集成 DialoGPT 预训练对话模型，提供 /chat 接口
"""

from flask import Flask, render_template, request, jsonify
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

app = Flask(__name__)

# ============================================================
# 模型配置
# ============================================================
MODEL_PATH = "/workspace/dialogpt_model"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = None
tokenizer = None


def load_model():
    """加载本地 DialoGPT 预训练对话模型"""
    global model, tokenizer

    print(f"[帧神AI] 正在加载模型: {MODEL_PATH}")
    print(f"[帧神AI] 设备: {DEVICE}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
    model.to(DEVICE)
    model.eval()
    tokenizer.pad_token = tokenizer.eos_token

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[帧神AI] 模型加载完成 | 参数量: {total_params:,}")


# ============================================================
# 对话生成
# ============================================================

def generate_reply(user_message: str) -> str:
    """使用 DialoGPT 生成回复"""
    if model is None or tokenizer is None:
        return "模型尚未加载完成，请稍后再试..."

    text = user_message.strip()
    if not text:
        return "你好像什么都没说呢，再说一遍？"

    input_text = text + tokenizer.eos_token

    with torch.no_grad():
        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(DEVICE)

        output_ids = model.generate(
            input_ids,
            max_length=input_ids.shape[1] + 60,
            min_length=input_ids.shape[1] + 3,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            top_p=0.92,
            top_k=50,
            temperature=0.8,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            early_stopping=True,
        )

        response_ids = output_ids[:, input_ids.shape[-1]:]
        reply = tokenizer.decode(response_ids[0], skip_special_tokens=True)

    reply = reply.strip()
    if not reply:
        return "我还在学习中，暂时不太理解你说的话，换个说法试试？"

    return reply


# ============================================================
# 路由
# ============================================================

@app.route("/")
def index():
    """渲染帧神AI聊天页面"""
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    """处理聊天请求"""
    data = request.get_json()
    if not data or "message" not in data:
        return jsonify({"error": "请提供 message 参数"}), 400

    user_message = data["message"]
    ai_reply = generate_reply(user_message)
    return jsonify({"reply": ai_reply})


# ============================================================
# 启动
# ============================================================

if __name__ == "__main__":
    load_model()
    print("[帧神AI] 服务启动 → http://127.0.0.1:5000")
    print("[帧神AI] 线上域名 → https://zhenshenai.com")
    app.run(debug=True, host="0.0.0.0", port=5000)
