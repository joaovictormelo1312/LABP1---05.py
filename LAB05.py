import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer

# =========================================================
# CONFIGURAÇÕES
# =========================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_SENTENCES = 1000
BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-4
D_MODEL = 128
NHEAD = 4
NUM_ENCODER_LAYERS = 2
NUM_DECODER_LAYERS = 2
DIM_FEEDFORWARD = 512
DROPOUT = 0.1
MAX_LEN = 64

SRC_LANG = "en"
TGT_LANG = "de"

# =========================================================
# TOKENIZADOR
# O PDF sugere usar um tokenizador pré-treinado do HF
# e adicionar tokens de início e fim na saída do decoder.
# =========================================================
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

SPECIAL_TOKENS = {
    "additional_special_tokens": ["<START>", "<EOS>"]
}
tokenizer.add_special_tokens(SPECIAL_TOKENS)

START_TOKEN_ID = tokenizer.convert_tokens_to_ids("<START>")
EOS_TOKEN_ID = tokenizer.convert_tokens_to_ids("<EOS>")
PAD_TOKEN_ID = tokenizer.pad_token_id

if PAD_TOKEN_ID is None:
    raise ValueError("O tokenizer precisa ter pad_token_id definido.")

VOCAB_SIZE = len(tokenizer)

# =========================================================
# DATASET
# O PDF sugere multi30k ou opus_books, usando um subset pequeno.
# =========================================================
def load_translation_pairs(num_sentences=1000):
    dataset = load_dataset("bentrevett/multi30k", split="train")

    src_texts = []
    tgt_texts = []

    count = 0
    for item in dataset:
        # Estrutura do multi30k:
        # item["translation"] -> {"de": "...", "en": "..."} em muitas versões
        if "translation" in item:
            translation = item["translation"]
            src = translation[SRC_LANG].strip()
            tgt = translation[TGT_LANG].strip()
        else:
            # fallback caso o formato venha diferente
            src = item[SRC_LANG].strip()
            tgt = item[TGT_LANG].strip()

        if src and tgt:
            src_texts.append(src)
            tgt_texts.append(tgt)
            count += 1

        if count >= num_sentences:
            break

    return src_texts, tgt_texts

# =========================================================
# TOKENIZAÇÃO E PREPARAÇÃO
# O PDF pede:
# - converter frases em IDs
# - adicionar <START> e <EOS> no decoder
# - padding para mesmo tamanho no batch
# =========================================================
def encode_source(text):
    encoded = tokenizer.encode(
        text,
        add_special_tokens=True,
        truncation=True,
        max_length=MAX_LEN
    )
    return encoded

def encode_target(text):
    # para a saída do decoder, adicionamos <START> e <EOS>
    tokens = tokenizer.encode(
        text,
        add_special_tokens=False,
        truncation=True,
        max_length=MAX_LEN - 2
    )
    tokens = [START_TOKEN_ID] + tokens + [EOS_TOKEN_ID]
    return tokens

class TranslationDataset(Dataset):
    def __init__(self, src_texts, tgt_texts):
        self.src_data = [encode_source(text) for text in src_texts]
        self.tgt_data = [encode_target(text) for text in tgt_texts]

    def __len__(self):
        return len(self.src_data)

    def __getitem__(self, idx):
        return self.src_data[idx], self.tgt_data[idx]

def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)

    src_max_len = min(max(len(x) for x in src_batch), MAX_LEN)
    tgt_max_len = min(max(len(x) for x in tgt_batch), MAX_LEN)

    padded_src = []
    padded_tgt = []

    for src_ids, tgt_ids in zip(src_batch, tgt_batch):
        src_ids = src_ids[:src_max_len]
        tgt_ids = tgt_ids[:tgt_max_len]

        src_pad_len = src_max_len - len(src_ids)
        tgt_pad_len = tgt_max_len - len(tgt_ids)

        padded_src.append(src_ids + [PAD_TOKEN_ID] * src_pad_len)
        padded_tgt.append(tgt_ids + [PAD_TOKEN_ID] * tgt_pad_len)

    src_tensor = torch.tensor(padded_src, dtype=torch.long)
    tgt_tensor = torch.tensor(padded_tgt, dtype=torch.long)

    return src_tensor, tgt_tensor

# =========================================================
# POSICIONAL ENCODING
# =========================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: [batch, seq_len, d_model]
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

# =========================================================
# MODELO TRANSFORMER
# Estrutura viável para o laboratório: d_model=128, h=4, N=2
# =========================================================
class Seq2SeqTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=128,
        nhead=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=512,
        dropout=0.1,
        pad_token_id=0
    ):
        super().__init__()
        self.d_model = d_model
        self.pad_token_id = pad_token_id

        self.src_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.tgt_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.positional_encoding = PositionalEncoding(d_model, dropout=dropout)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        self.fc_out = nn.Linear(d_model, vocab_size)

    def generate_square_subsequent_mask(self, sz, device):
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()
        return mask

    def make_padding_mask(self, x):
        return x == self.pad_token_id

    def forward(self, src, tgt_input):
        # src: [batch, src_len]
        # tgt_input: [batch, tgt_len]

        src_padding_mask = self.make_padding_mask(src)
        tgt_padding_mask = self.make_padding_mask(tgt_input)
        tgt_mask = self.generate_square_subsequent_mask(tgt_input.size(1), tgt_input.device)

        src_emb = self.src_embedding(src) * math.sqrt(self.d_model)
        tgt_emb = self.tgt_embedding(tgt_input) * math.sqrt(self.d_model)

        src_emb = self.positional_encoding(src_emb)
        tgt_emb = self.positional_encoding(tgt_emb)

        output = self.transformer(
            src=src_emb,
            tgt=tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask
        )

        logits = self.fc_out(output)  # [batch, tgt_len, vocab_size]
        return logits

# =========================================================
# FUNÇÃO DE TREINO
# O PDF pede CrossEntropyLoss com ignore_index no padding,
# Adam e impressão do Loss por época.
# =========================================================
def train_model(model, dataloader, optimizer, criterion, epochs):
    model.train()

    for epoch in range(1, epochs + 1):
        total_loss = 0.0

        for src, tgt in dataloader:
            src = src.to(DEVICE)
            tgt = tgt.to(DEVICE)

            # Teacher forcing:
            # entrada do decoder = tudo menos o último token
            # target real = tudo menos o primeiro token
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            optimizer.zero_grad()

            logits = model(src, tgt_input)
            # logits: [batch, tgt_len, vocab]
            # CrossEntropyLoss espera [N, C] e target [N]
            logits = logits.reshape(-1, logits.size(-1))
            tgt_output = tgt_output.reshape(-1)

            loss = criterion(logits, tgt_output)
            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch}/{epochs} - Loss: {avg_loss:.4f}")

# =========================================================
# GERAÇÃO AUTO-REGRESSIVA
# Para o teste final de overfitting
# =========================================================
@torch.no_grad()
def greedy_decode(model, src_sentence, max_len=MAX_LEN):
    model.eval()

    src_ids = encode_source(src_sentence)
    src_tensor = torch.tensor([src_ids], dtype=torch.long).to(DEVICE)

    generated = [START_TOKEN_ID]

    for _ in range(max_len - 1):
        tgt_tensor = torch.tensor([generated], dtype=torch.long).to(DEVICE)

        logits = model(src_tensor, tgt_tensor)
        next_token_id = logits[:, -1, :].argmax(dim=-1).item()

        generated.append(next_token_id)

        if next_token_id == EOS_TOKEN_ID:
            break

    # remove <START> e corta no <EOS>
    decoded_ids = []
    for token_id in generated[1:]:
        if token_id == EOS_TOKEN_ID:
            break
        decoded_ids.append(token_id)

    return tokenizer.decode(decoded_ids, skip_special_tokens=True)

# =========================================================
# MAIN
# =========================================================
def main():
    print("Dispositivo:", DEVICE)
    print("Carregando dataset...")
    src_texts, tgt_texts = load_translation_pairs(NUM_SENTENCES)

    print(f"Quantidade de pares carregados: {len(src_texts)}")

    dataset = TranslationDataset(src_texts, tgt_texts)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn
    )

    model = Seq2SeqTransformer(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT,
        pad_token_id=PAD_TOKEN_ID
    ).to(DEVICE)

    # ajusta embeddings para novos tokens especiais
    model.src_embedding = nn.Embedding(VOCAB_SIZE, D_MODEL, padding_idx=PAD_TOKEN_ID).to(DEVICE)
    model.tgt_embedding = nn.Embedding(VOCAB_SIZE, D_MODEL, padding_idx=PAD_TOKEN_ID).to(DEVICE)
    model.fc_out = nn.Linear(D_MODEL, VOCAB_SIZE).to(DEVICE)

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN_ID)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print("\nIniciando treinamento...")
    train_model(model, dataloader, optimizer, criterion, EPOCHS)

    print("\n=== TESTE DE OVERFITTING ===")
    test_idx = 0
    src_example = src_texts[test_idx]
    tgt_example = tgt_texts[test_idx]

    prediction = greedy_decode(model, src_example)

    print(f"Frase origem ({SRC_LANG}): {src_example}")
    print(f"Tradução real ({TGT_LANG}): {tgt_example}")
    print(f"Tradução gerada: {prediction}")

if __name__ == "__main__":
    main()
