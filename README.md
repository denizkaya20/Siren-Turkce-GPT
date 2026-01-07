cat <<EOF > README.md
# 🚨 SIREN AI: Native Turkish GPT

SIREN AI is a specialized Turkish Large Language Model (LLM) trained from scratch. This project includes the complete pipeline for pre-training, supervised fine-tuning (SFT), and an RLHF-ready inference interface.

The model demonstrates strong capabilities in Turkish language understanding, instruction following, and cultural context awareness.

## 🌟 Features

* **Native Turkish Training:** Trained on the Cosmos Corpus and custom datasets.
* **SFT (Supervised Fine-Tuning):** Fine-tuned using Turkish Alpaca and encyclopedic datasets for instruction following.
* **RLHF Ready:** The inference interface includes a "Teacher Mode" to collect human feedback (DPO/RLHF) for future improvements.
* **Optimized Inference:** Uses \`torch.compile\` and quantization friendly architecture.

## 📂 Project Structure

* **\`app.py\`**: The main Gradio interface. It automatically downloads the model from Hugging Face and allows chatting + feedback collection.
* **\`train_runpod.py\`**: The script used for **Pre-training** (Cosmos stage).
* **\`sft_runpod.py\`**: The script used for **Supervised Fine-Tuning** (Alpaca + Knowledge injection).
* **\`prepare_cosmos.py\`**: Data processing script to prepare the Cosmos corpus.
* **\`binarize_data.py\`**: Script to convert text datasets into binary format for efficient training.
* **\`modeling_siren.py\`**: Custom GPT architecture definition.

## 🚀 Installation & Usage

1. **Clone the repository:**
   \`\`\`bash
   git clone https://github.com/YOUR_USERNAME/SIREN_Turkish_GPT.git
   cd SIREN_Turkish_GPT
   \`\`\`

2. **Install dependencies:**
   \`\`\`bash
   pip install -r requirements.txt
   \`\`\`

3. **Run the Application:**
   \`\`\`bash
   python app.py
   \`\`\`
   *The application will automatically download the necessary model weights (~1.3GB) from Hugging Face on the first run.*

## 🧠 Model Details

* **Architecture:** Custom GPT (Decoder-only Transformer)
* **Tokenizer:** Custom SentencePiece Unigram Model (32k Vocab)
* **Context Window:** 1024 Tokens
* **Hugging Face Repo:** [denizkaya2022/Siren_Turkce_GPT](https://huggingface.co/denizkaya2022/Siren_Turkce_GPT)

## 🤝 Contributing

Contributions are welcome! If you want to improve the dataset or the training logic, feel free to submit a Pull Request.

## 📜 License

MIT License.
EOF
