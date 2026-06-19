# Karthikeyan Kumaravel Krishnan
# Chain of Thought Reasoning with Mini Glossary Approach

# Standard libraries for parsing, file handling, and regex processing
import re
import os

# Data handling
import pandas as pd
import matplotlib.pyplot as plotMaker

# Embedding model to determine semantic similarity
from sentence_transformers import SentenceTransformer

# LLM API Client (Cerebras hosted LLaMA model)
from openai import OpenAI

# Evaluation metrics
from bert_score import score as bert_score
from sklearn.metrics.pairwise import cosine_similarity

# Configuring Cerebras API
client = OpenAI(
    api_key="ENTER API KEY HERE",
    base_url="https://api.cerebras.ai/v1"
)
MODEL = "llama3.1-8b"

# Results will be saved in an "outputs" folder which will contain CSV results and plots
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Function used to load a txt file that contains multiple choice question/answers for any given language
def load_mcq_txt(path):
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    blocks = re.split(r'(?=Question \d+:)', content.strip())
    data = []

    # Parses through the file and extracts the multiple choice questions, options, and correct answers
    for block in blocks:
        if not block.strip():
            continue

        lines = [l.strip() for l in block.split("\n") if l.strip()]

        question_lines = []
        options = {}
        answer = None

        for line in lines:
            if re.match(r'^[A-D]:', line):
                options[line[0]] = line[2:].strip()
            elif line.startswith("Correct Answer:"):
                answer = line.split(":", 1)[1].strip().strip('`')
            else:
                question_lines.append(line)

        question = "\n".join(question_lines).strip()

        if question and options and answer:
            data.append({
                "question": question,
                "options": options,
                "answer": answer
            })

    print(f"MCQ loaded: {len(data)} samples")
    return data

# Function used to load a txt file that contains interlinear glossed text and its translation for any given language
def load_igt_txt(path):
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()

    blocks = raw.split("\\label")
    data = []

    # Parses through the file and extracts the morphemes, glosses, and full sentence translations
    for block in blocks:
        word_line_matches = re.findall(r"\\gll (.*?)\\gls", block, re.DOTALL)
        gloss_line_matches = re.findall(r"\\gls (.*?)\\glt", block, re.DOTALL)
        translation_line_matches = re.findall(r"\\glt\s*`?(.*?)'?\s*$", block, re.DOTALL)

        if not word_line_matches or not gloss_line_matches or not translation_line_matches:
            continue

        data.append({
            "words": word_line_matches[0].strip().split(),
            "glosses": gloss_line_matches[0].strip().split(),
            "translation": translation_line_matches[0].strip()
        })

    print(f"IGT loaded: {len(data)} samples")
    return data

# Function builds the prompt for LLM to answer multiple choice questions
def build_mcq_prompt(item, language):
    options_dict = item["options"]

    options_text = "\n".join(
        f"{option_label}: {option_text}"
        for option_label, option_text in options_dict.items()
    )

    return f""" Answer the following multiple choice question about {language} grammar.

    Respond ONLY with A, B, C, or D.

    Question:
    {item["question"]}

    Options:
    {options_text}

    Answer:
    """

# Function builds a open-ended CoT prompt for LLM to predict the correct translated sentence
def build_cot_prompt(item, language):
    morpheme_sentence = " ".join(item["words"])
    glossary = "\n".join(
        f"{word} → {gloss}" for word, gloss in zip(item["words"], item["glosses"])
    )
    return f"""You are a professional linguist.

    You will translate a sentence using step-by-step reasoning.

    Follow these steps internally:
    1. Identify each morpheme and its meaning using the glossary.
    2. Understand how the meanings combine into a full sentence.
    3. Produce a fluent natural English translation.

    IMPORTANT:
    Do NOT show your reasoning.
    Return ONLY the final translation.

    Language: {language}
    Sentence: {morpheme_sentence}

    Gloss:
    {glossary}

    Final Answer:
    """

# Function used to send the prompts to the LLM
def call_llm(prompt):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=150,
    )
    model_output = response.choices[0].message.content.strip()

    return model_output

# Used to determine if LLM's predictions were correct or not by comparing answer choices
def extract_mcq(text):
    match = re.search(r'\b([A-D])\b', text)
    return match.group(1) if match else "?"

# Used to determine how close the LLM was correct to the actual translation using cosine similarity
def cosine_score(embedding_model, predicted_text, gold_truth_text):
    predicted_embedding = embedding_model.encode(str(predicted_text))
    gold_embedding = embedding_model.encode(str(gold_truth_text))
    return float(cosine_similarity([predicted_embedding], [gold_embedding])[0][0])

# Evaluates Multiple Choice Question performance by checking how many questions it got right (accuracy)
def evaluate_mcq(data, language):
    print("\n========== MCQ EVALUATION ==========\n")

    rows = []
    correct_list = []

    for sample_index, item in enumerate(data):
        prompt_text = build_mcq_prompt(item, language)
        output = call_llm(prompt_text)

        predicted_option = extract_mcq(output)
        gold_truth_option = item["answer"]
        correct_flag = predicted_option == gold_truth_option

        correct_list.append(correct_flag)

        running_acc = sum(correct_list) / len(correct_list)

        rows.append({
            "id": sample_index,
            "prompt": prompt_text,
            "prediction": predicted_option,
            "gold": gold_truth_option,
            "correct": correct_flag,
            "running_accuracy": running_acc
        })

    mcq_results_df = pd.DataFrame(rows)
    print(f"Final MCQ Accuracy: {mcq_results_df['running_accuracy'].iloc[-1]:.4f}")
    return mcq_results_df

# Evaluates Chain-Of-Thought translation performance by computing cosine similarity and BERTScore to see how similar its prediction was to the actual translated text.
def evaluate_cot(data, embed_model, language):
    print("\n========== CoT EVALUATION ==========\n")

    rows = []
    predicted_translations = []
    gold_translations = []

    for sample_index, item in enumerate(data):
        prompt_text = build_cot_prompt(item, language)
        model_output = call_llm(prompt_text)

        predicted_translation = model_output
        gold_translation = item["translation"]

        cosine_similarity_score = cosine_score(embed_model, predicted_translation, gold_translation)

        predicted_translations.append(predicted_translation)
        gold_translations.append(gold_translation)

        rows.append({
            "id": sample_index,
            "prompt": prompt_text,
            "prediction": predicted_translation,
            "gold": gold_translation,
            "cosine": cosine_similarity_score
        })

    # Calculates BERTScore
    precision_scores, recall_scores, F1_scores = bert_score(
        predicted_translations,
        gold_translations,
        lang="en",
        verbose=False
    )

    bert_f1_scores = F1_scores.tolist()

    for sample_index in range(len(rows)):
        rows[sample_index]["bertscore"] = bert_f1_scores[sample_index]

    cot_results_df = pd.DataFrame(rows)

    cot_results_df["cosine_mean"] = cot_results_df["cosine"].mean()
    cot_results_df["bert_mean"] = cot_results_df["bertscore"].mean()

    print(f"Final Cosine Score: {cot_results_df['cosine'].mean():.4f}")
    print(f"Final BERTScore F1: {cot_results_df['bertscore'].mean():.4f}")

    return cot_results_df

# Plotting results
def plot_evaluation_trends(mcq_df, cot_df, language):

    plotMaker.figure(figsize=(7, 4.5))

    metrics = ["MCQ Accuracy", "CoT (Cosine)", "CoT (BERTScore)"]

    values = [
        mcq_df["running_accuracy"].iloc[-1],
        cot_df["cosine"].mean(),
        cot_df["bertscore"].mean()
    ]

    colors = ["#d62728", "#9467bd", "#ffbf00"]
    bars = plotMaker.bar(metrics, values, color=colors)

    plotMaker.xlabel("Number of Samples")
    plotMaker.ylabel("Accuracy / Cosine Similarity")

    plotMaker.title(f"{language}: Evaluation Trends (MCQ vs CoT)")

    plotMaker.ylim(0, 1)

    plotMaker.grid(axis="y", alpha=0.2, linestyle="--", linewidth=0.6)

    plotMaker.legend(
        bars,
        ["MCQ", "CoT (Cosine)", "CoT (BERTScore)"],
        loc="lower right",
        frameon=True,
        fancybox=True,
        framealpha=0.9,
        edgecolor="0.3",
        fontsize=9,
        title="Legend"
    )

    plotMaker.xticks(fontsize=9)
    plotMaker.yticks(fontsize=9)

    plotMaker.tight_layout(pad=0.3)

    plotMaker.savefig(
        os.path.join(OUTPUT_DIR, f"{language.lower()}_evaluation_trends.png"),
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.05
    )

    plotMaker.close()

def main():

    embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    datasets = [
        {
            "language": "Mauwake",
            "mcq_file": "mauwake_mcq.txt",
            "igt_file": "mauwake_igt.txt"
        },
        {
            "language": "Kalamang",
            "mcq_file": "kalamang_mcq.txt",
            "igt_file": "kalamang_igt.txt"
        },
        {
            "language": "Moloko",
            "mcq_file": "moloko_mcq.txt",
            "igt_file": "moloko_igt.txt"
        }
    ]

    all_summary = []

    for dataset in datasets:
        language = dataset["language"]

        print("\n" + "="*60)
        print(f"RUNNING LANGUAGE: {language}")
        print("="*60)

        mcq_data = load_mcq_txt(dataset["mcq_file"])
        igt_data = load_igt_txt(dataset["igt_file"])

        mcq_df = evaluate_mcq(mcq_data, language)
        cot_df = evaluate_cot(igt_data, embed_model, language)

        # Save per-language results
        mcq_df.to_csv(os.path.join(OUTPUT_DIR, f"{language.lower()}_mcq_results.csv"), index=False)
        cot_df.to_csv(os.path.join(OUTPUT_DIR, f"{language.lower()}_cot_results.csv"), index=False)

        plot_evaluation_trends(mcq_df, cot_df, language)

        all_summary.append({
            "language": language,
            "mcq_accuracy": mcq_df["running_accuracy"].iloc[-1],
            "cot_cosine": cot_df["cosine"].mean(),
            "cot_bertscore": cot_df["bertscore"].mean()
        })

    # Save summary table for ALL languages
    summary_df = pd.DataFrame(all_summary)
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "language_summary.csv"), index=False)

    print("\nDONE ALL LANGUAGES")
    print(summary_df)

if __name__ == "__main__":
    main()
