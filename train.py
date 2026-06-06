"""
Seq2Seq + Bahdanau Attention 对话模型训练脚本（改进版）
- 支持从检查点恢复训练
- 使用全部数据 + 更好的超参数
- Beam Search 推理
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import jieba
import numpy as np
import os
import re
import pickle
import random
from collections import Counter

# ============================================================
# 超参数
# ============================================================
HIDDEN_SIZE = 128
EMBEDDING_SIZE = 128
NUM_LAYERS = 1
DROPOUT = 0.2
MAX_LENGTH = 30
BATCH_SIZE = 256
EPOCHS = 8
LEARNING_RATE = 0.003
TEACHER_FORCING_RATIO = 0.5
VOCAB_SIZE = 6000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "/workspace/chatbot_model.pt"
CHECKPOINT_DIR = "/workspace"

print(f"使用设备: {DEVICE}")

# ============================================================
# 数据加载与解析
# ============================================================

def parse_dialogue_file(filepath):
    """解析 XDailyDialog 对话文件，提取 (上句, 下句) 对"""
    pairs = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 1:
                continue
            dialogue_text = parts[0]
            turns = dialogue_text.split("__eou__")
            turns = [t.strip() for t in turns if t.strip()]
            for i in range(len(turns) - 1):
                src = turns[i]
                tgt = turns[i + 1]
                if len(src) > 2 and len(tgt) > 2:
                    pairs.append((src, tgt))
    return pairs


def load_data():
    """加载中英双语对话数据"""
    zh_train = "/data/user/work/XDailyDialog/data/zh_train_human.txt"
    en_train = "/data/user/work/XDailyDialog/data/en_train_human.txt"
    zh_dev = "/data/user/work/XDailyDialog/data/zh_dev_human.txt"
    en_dev = "/data/user/work/XDailyDialog/data/en_dev_human.txt"

    print("加载中文训练对话数据...")
    zh_train_pairs = parse_dialogue_file(zh_train)
    print(f"  中文训练对话对: {len(zh_train_pairs)}")

    print("加载英文训练对话数据...")
    en_train_pairs = parse_dialogue_file(en_train)
    print(f"  英文训练对话对: {len(en_train_pairs)}")

    print("加载中文验证对话数据...")
    zh_dev_pairs = parse_dialogue_file(zh_dev)
    print(f"  中文验证对话对: {len(zh_dev_pairs)}")

    print("加载英文验证对话数据...")
    en_dev_pairs = parse_dialogue_file(en_dev)
    print(f"  英文验证对话对: {len(en_dev_pairs)}")

    train_pairs = zh_train_pairs + en_train_pairs
    val_pairs = zh_dev_pairs + en_dev_pairs
    random.shuffle(train_pairs)
    random.shuffle(val_pairs)

    print(f"  总训练对: {len(train_pairs)}, 总验证对: {len(val_pairs)}")
    return train_pairs, val_pairs


# ============================================================
# 分词器
# ============================================================

class BilingualTokenizer:
    """中英混合分词器"""

    def __init__(self):
        self.word2idx = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.idx2word = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.vocab_size = 4

    def tokenize(self, text):
        tokens = []
        segments = re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z]+|[0-9]+|[^\s\w\u4e00-\u9fff]', text)
        for seg in segments:
            if re.match(r'[\u4e00-\u9fff]+', seg):
                tokens.extend(jieba.lcut(seg))
            else:
                tokens.append(seg.lower())
        return tokens

    def build_vocab(self, texts, max_vocab=VOCAB_SIZE):
        counter = Counter()
        for text in texts:
            tokens = self.tokenize(text)
            counter.update(tokens)
        most_common = counter.most_common(max_vocab - 4)
        for word, _ in most_common:
            self.word2idx[word] = self.vocab_size
            self.idx2word[self.vocab_size] = word
            self.vocab_size += 1
        print(f"词表大小: {self.vocab_size}")

    def encode(self, text, max_len=MAX_LENGTH):
        tokens = self.tokenize(text)
        indices = [self.word2idx.get(t, self.word2idx["<UNK>"]) for t in tokens]
        indices = indices[:max_len - 1]
        indices.append(self.word2idx["<EOS>"])
        return indices

    def decode(self, indices, skip_special=True):
        words = []
        for idx in indices:
            if skip_special and idx in [0, 1, 2, 3]:
                if idx == 2:
                    break
                continue
            word = self.idx2word.get(idx, "<UNK>")
            words.append(word)
        return "".join(words)


# ============================================================
# 数据集
# ============================================================

class ChatDataset(Dataset):
    def __init__(self, pairs, tokenizer, max_len=MAX_LENGTH):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src_text, tgt_text = self.pairs[idx]
        src_indices = self.tokenizer.encode(src_text, self.max_len)
        tgt_indices = self.tokenizer.encode(tgt_text, self.max_len)
        return (
            torch.tensor(src_indices, dtype=torch.long),
            torch.tensor(tgt_indices, dtype=torch.long),
        )


def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_lens = [len(s) for s in src_batch]
    tgt_lens = [len(t) for t in tgt_batch]
    src_padded = nn.utils.rnn.pad_sequence(src_batch, batch_first=True, padding_value=0)
    tgt_padded = nn.utils.rnn.pad_sequence(tgt_batch, batch_first=True, padding_value=0)
    return src_padded, torch.tensor(src_lens), tgt_padded, torch.tensor(tgt_lens)


# ============================================================
# 模型组件
# ============================================================

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.Wa = nn.Linear(hidden_size * 2, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, decoder_hidden, encoder_outputs):
        decoder_hidden = decoder_hidden.unsqueeze(1)
        scores = self.Va(torch.tanh(self.Wa(encoder_outputs) + self.Ua(decoder_hidden)))
        scores = scores.squeeze(2)
        attn_weights = F.softmax(scores, dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
        context = context.squeeze(1)
        return context, attn_weights


class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
        self.gru = nn.GRU(
            embedding_size, hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, lengths):
        embedded = self.dropout(self.embedding(x))
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_outputs, hidden = self.gru(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)
        return outputs, hidden


class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
        self.attention = BahdanauAttention(hidden_size)
        self.gru = nn.GRU(
            embedding_size + hidden_size * 2, hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, hidden, encoder_outputs):
        embedded = self.dropout(self.embedding(x))
        query = hidden[-1]
        context, attn_weights = self.attention(query, encoder_outputs)
        context = context.unsqueeze(1)
        gru_input = torch.cat([embedded, context], dim=2)
        output, hidden = self.gru(gru_input, hidden)
        output = output.squeeze(1)
        prediction = self.fc(output)
        return prediction, hidden, attn_weights


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, src_lens, tgt, teacher_forcing_ratio=0.5):
        batch_size = src.size(0)
        tgt_len = tgt.size(1)
        tgt_vocab_size = self.decoder.fc.out_features
        outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(self.device)

        encoder_outputs, encoder_hidden = self.encoder(src, src_lens)

        num_layers = encoder_hidden.size(0) // 2
        decoder_hidden = encoder_hidden.view(num_layers, 2, batch_size, -1)
        decoder_hidden = decoder_hidden.sum(dim=1)

        decoder_input = torch.full((batch_size, 1), 1, dtype=torch.long).to(self.device)

        for t in range(tgt_len):
            prediction, decoder_hidden, _ = self.decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            outputs[:, t, :] = prediction
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = prediction.argmax(1).unsqueeze(1)
            decoder_input = tgt[:, t].unsqueeze(1) if teacher_force else top1

        return outputs


# ============================================================
# Beam Search 推理
# ============================================================

def generate_response_beam(model, tokenizer, input_text, beam_width=3, max_len=MAX_LENGTH, device=DEVICE):
    """使用 Beam Search 生成回复"""
    model.eval()
    with torch.no_grad():
        src_indices = tokenizer.encode(input_text, max_len)
        src_tensor = torch.tensor([src_indices], dtype=torch.long).to(device)
        src_lengths = torch.tensor([len(src_indices)], dtype=torch.long)

        encoder_outputs, encoder_hidden = model.encoder(src_tensor, src_lengths)

        num_layers = encoder_hidden.size(0) // 2
        decoder_hidden = encoder_hidden.view(num_layers, 2, 1, -1).sum(dim=1)

        # Beam: (score, token_sequence, hidden_state)
        beams = [(0.0, [1], decoder_hidden)]  # (score, tokens, hidden)

        for _ in range(max_len):
            all_candidates = []
            for score, tokens, hidden in beams:
                if tokens[-1] == 2:  # EOS
                    all_candidates.append((score, tokens, hidden))
                    continue
                decoder_input = torch.tensor([[tokens[-1]]], dtype=torch.long).to(device)
                prediction, new_hidden, _ = model.decoder(
                    decoder_input, hidden, encoder_outputs
                )
                log_probs = F.log_softmax(prediction[0], dim=0)
                topk_probs, topk_ids = torch.topk(log_probs, beam_width)
                for i in range(beam_width):
                    new_score = score + topk_probs[i].item()
                    new_tokens = tokens + [topk_ids[i].item()]
                    all_candidates.append((new_score, new_tokens, new_hidden))

            # 保留 top-k
            beams = sorted(all_candidates, key=lambda x: x[0], reverse=True)[:beam_width]

            # 检查是否所有 beam 都以 EOS 结束
            if all(b[1][-1] == 2 for b in beams):
                break

        # 选择最佳 beam
        best_beam = beams[0]
        tokens = best_beam[1]
        return tokenizer.decode(tokens)


# ============================================================
# 训练
# ============================================================

def train_epoch(model, dataloader, optimizer, criterion, epoch):
    model.train()
    total_loss = 0
    for batch_idx, (src, src_lens, tgt, tgt_lens) in enumerate(dataloader):
        src, tgt = src.to(DEVICE), tgt.to(DEVICE)
        src_lens = src_lens.to(DEVICE)

        optimizer.zero_grad()
        output = model(src, src_lens, tgt, TEACHER_FORCING_RATIO)
        output = output[:, 1:].reshape(-1, output.size(-1))
        tgt = tgt[:, 1:].reshape(-1)

        loss = criterion(output, tgt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

        if (batch_idx + 1) % 100 == 0:
            print(f"  Epoch {epoch}, Batch {batch_idx+1}/{len(dataloader)}, Loss: {loss.item():.4f}")

    return total_loss / len(dataloader)


def validate(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for src, src_lens, tgt, tgt_lens in dataloader:
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)
            src_lens = src_lens.to(DEVICE)
            output = model(src, src_lens, tgt, teacher_forcing_ratio=0.0)
            output = output[:, 1:].reshape(-1, output.size(-1))
            tgt = tgt[:, 1:].reshape(-1)
            loss = criterion(output, tgt)
            total_loss += loss.item()
    return total_loss / len(dataloader)


def save_checkpoint(model, tokenizer, epoch, val_loss, best_val_loss, path):
    torch.save({
        "model_state_dict": model.state_dict(),
        "tokenizer": tokenizer,
        "vocab_size": tokenizer.vocab_size,
        "hidden_size": HIDDEN_SIZE,
        "embedding_size": EMBEDDING_SIZE,
        "num_layers": NUM_LAYERS,
        "max_length": MAX_LENGTH,
        "epoch": epoch,
        "val_loss": val_loss,
    }, path)
    print(f"  >> 保存模型 (Epoch {epoch}, Val Loss: {val_loss:.4f})")


# ============================================================
# 主流程
# ============================================================

def main():
    print("=" * 60)
    print("Seq2Seq + Attention 对话模型训练（改进版）")
    print("=" * 60)

    # 1. 加载数据
    print("\n[1/5] 加载数据...")
    train_pairs, val_pairs = load_data()

    # 2. 构建分词器
    print("\n[2/5] 构建分词器...")
    all_texts = [p[0] for p in train_pairs] + [p[1] for p in train_pairs]
    tokenizer = BilingualTokenizer()
    tokenizer.build_vocab(all_texts)

    # 3. 创建数据集
    print("\n[3/5] 构建数据加载器...")
    train_dataset = ChatDataset(train_pairs, tokenizer)
    val_dataset = ChatDataset(val_pairs, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=0)

    # 4. 构建模型
    print("\n[4/5] 构建模型...")
    encoder = Encoder(tokenizer.vocab_size, EMBEDDING_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT)
    decoder = Decoder(tokenizer.vocab_size, EMBEDDING_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT)
    model = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  总参数量: {total_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-5)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    best_val_loss = float("inf")
    start_epoch = 1

    # 5. 训练
    print(f"\n[5/5] 开始训练 ({EPOCHS} epochs)...")
    for epoch in range(start_epoch, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, epoch)
        val_loss = validate(model, val_loader, criterion)
        scheduler.step(val_loss)

        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:2d}: Train={train_loss:.4f}, Val={val_loss:.4f}, LR={lr:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, tokenizer, epoch, val_loss, best_val_loss, MODEL_PATH)

    # 6. 测试
    print(f"\n训练完成! 最佳验证损失: {best_val_loss:.4f}")
    print("\n模型测试 (Beam Search):")
    test_inputs = [
        "你好",
        "Hello, how are you?",
        "你喜欢什么？",
        "What do you like to do?",
        "谢谢你的帮助",
        "Thank you very much",
        "今天天气真好",
        "It's a beautiful day",
    ]
    for text in test_inputs:
        reply = generate_response_beam(model, tokenizer, text, beam_width=3)
        print(f"  Q: {text}")
        print(f"  A: {reply}")
        print()

    print("模型已保存到", MODEL_PATH)


if __name__ == "__main__":
    main()
