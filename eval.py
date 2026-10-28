import time
import gradio as gr
import pandas as pd
from collections import defaultdict
from dotenv import load_dotenv

from testing.evaluations import evaluate_all_retrieval, evaluate_all_answers
from testing.test import list_test_sets
from providers import EMBEDDING_MODELS, CHAT_MODELS, embeddings_for

load_dotenv(override=True)

MRR_GREEN = 0.9
MRR_AMBER = 0.75
NDCG_GREEN = 0.9
NDCG_AMBER = 0.75
COVERAGE_GREEN = 90.0
COVERAGE_AMBER = 75.0

ANSWER_GREEN = 4.5
ANSWER_AMBER = 4.0


def get_color(value, metric_type):
    """Get color based on metric value and type."""
    if metric_type == "mrr":
        if value >= MRR_GREEN:
            return "green"
        elif value >= MRR_AMBER:
            return "orange"
        else:
            return "red"
    elif metric_type == "ndcg":
        if value >= NDCG_GREEN:
            return "green"
        elif value >= NDCG_AMBER:
            return "orange"
        else:
            return "red"
    elif metric_type == "coverage":
        if value >= COVERAGE_GREEN:
            return "green"
        elif value >= COVERAGE_AMBER:
            return "orange"
        else:
            return "red"
    elif metric_type in ["accuracy", "completeness", "relevance"]:
        if value >= ANSWER_GREEN:
            return "green"
        elif value >= ANSWER_AMBER:
            return "orange"
        else:
            return "red"
    return "black"


def format_metric_html(label, value, metric_type, is_percentage=False, score_format=False):
    """Format a metric with color coding."""
    color = get_color(value, metric_type)
    if is_percentage:
        value_str = f"{value:.1f}%"
    elif score_format:
        value_str = f"{value:.2f}/5"
    else:
        value_str = f"{value:.4f}"
    return f"""
    <div style="margin: 10px 0; padding: 15px; background-color: #f5f5f5; border-radius: 8px; border-left: 5px solid {color};">
        <div style="font-size: 14px; color: #666; margin-bottom: 5px;">{label}</div>
        <div style="font-size: 28px; font-weight: bold; color: {color};">{value_str}</div>
    </div>
    """


retrieval_running = False
retrieval_paused = False
answer_running = False
answer_paused = False


def run_retrieval_evaluation(test_set, embedding_model, chat_model, progress=gr.Progress()):
    global retrieval_running, retrieval_paused

    if retrieval_running:
        retrieval_paused = not retrieval_paused
        return gr.skip(), gr.skip()

    retrieval_running = True
    retrieval_paused = False

    embeddings = embeddings_for(embedding_model)

    total_mrr = 0.0
    total_ndcg = 0.0
    total_coverage = 0.0
    category_mrr = defaultdict(list)
    count = 0

    for test, result, prog_value in evaluate_all_retrieval(test_set, embeddings, embedding_model, chat_model):
        while retrieval_paused:
            progress(prog_value, desc="Paused - click Run Evaluation again to resume")
            time.sleep(0.5)
        count += 1
        total_mrr += result.mrr
        total_ndcg += result.ndcg
        total_coverage += result.keyword_coverage
        category_mrr[test.category].append(result.mrr)
        progress(prog_value, desc=f"Evaluating test {count}...")

    retrieval_running = False

    avg_mrr = total_mrr / count
    avg_ndcg = total_ndcg / count
    avg_coverage = total_coverage / count

    final_html = f"""
    <div style="padding: 0;">
        {format_metric_html("Mean Reciprocal Rank (MRR)", avg_mrr, "mrr")}
        {format_metric_html("Normalized DCG (nDCG)", avg_ndcg, "ndcg")}
        {format_metric_html("Keyword Coverage", avg_coverage, "coverage", is_percentage=True)}
        <div style="margin-top: 20px; padding: 10px; background-color: #d4edda; border-radius: 5px; text-align: center; border: 1px solid #c3e6cb;">
            <span style="font-size: 14px; color: #155724; font-weight: bold;">✓ Evaluation Complete: {count} tests</span>
        </div>
    </div>
    """

    category_data = [
        {"Category": category, "Average MRR": sum(scores) / len(scores)}
        for category, scores in category_mrr.items()
    ]
    df = pd.DataFrame(category_data)

    return final_html, df


def run_answer_evaluation(test_set, embedding_model, chat_model, progress=gr.Progress()):
    global answer_running, answer_paused

    if answer_running:
        answer_paused = not answer_paused
        return gr.skip(), gr.skip()

    answer_running = True
    answer_paused = False

    embeddings = embeddings_for(embedding_model)

    total_accuracy = 0.0
    total_completeness = 0.0
    total_relevance = 0.0
    category_accuracy = defaultdict(list)
    count = 0

    for test, result, prog_value in evaluate_all_answers(test_set, embeddings, embedding_model, chat_model):
        while answer_paused:
            progress(prog_value, desc="Paused - click Run Evaluation again to resume")
            time.sleep(0.5)
        count += 1
        total_accuracy += result.accuracy
        total_completeness += result.completeness
        total_relevance += result.relevance
        category_accuracy[test.category].append(result.accuracy)
        progress(prog_value, desc=f"Evaluating test {count}...")

    answer_running = False

    avg_accuracy = total_accuracy / count
    avg_completeness = total_completeness / count
    avg_relevance = total_relevance / count

    final_html = f"""
    <div style="padding: 0;">
        {format_metric_html("Accuracy", avg_accuracy, "accuracy", score_format=True)}
        {format_metric_html("Completeness", avg_completeness, "completeness", score_format=True)}
        {format_metric_html("Relevance", avg_relevance, "relevance", score_format=True)}
        <div style="margin-top: 20px; padding: 10px; background-color: #d4edda; border-radius: 5px; text-align: center; border: 1px solid #c3e6cb;">
            <span style="font-size: 14px; color: #155724; font-weight: bold;">✓ Evaluation Complete: {count} tests</span>
        </div>
    </div>
    """

    category_data = [
        {"Category": category, "Average Accuracy": sum(scores) / len(scores)}
        for category, scores in category_accuracy.items()
    ]
    df = pd.DataFrame(category_data)

    return final_html, df


def build_eval_tab():
    """Build the evaluation dashboard UI inside the caller's gr.Blocks context."""
    gr.Markdown("# 📊 RAG Evaluation Dashboard")

    test_sets = list_test_sets()
    with gr.Row():
        test_set = gr.Dropdown(choices=test_sets, value=test_sets[0] if test_sets else None, label="Test Set")
        embedding_model = gr.Dropdown(choices=EMBEDDING_MODELS, value=EMBEDDING_MODELS[0], label="Embedding Model")
        chat_model = gr.Dropdown(choices=CHAT_MODELS, value=CHAT_MODELS[0], label="Chat Model")

    gr.Markdown("## 🔍 Retrieval Evaluation")
    retrieval_button = gr.Button("Run Evaluation", variant="primary", size="lg")

    with gr.Row():
        with gr.Column(scale=1):
            retrieval_metrics = gr.HTML(
                "<div style='padding: 20px; text-align: center; color: #999;'>Click 'Run Evaluation' to start</div>"
            )
        with gr.Column(scale=1):
            retrieval_chart = gr.BarPlot(
                x="Category", y="Average MRR", title="Average MRR by Category", y_lim=[0, 1], height=450, x_label_angle=-40
            )

    gr.Markdown("## 💬 Answer Evaluation")
    answer_button = gr.Button("Run Evaluation", variant="primary", size="lg")

    with gr.Row():
        with gr.Column(scale=1):
            answer_metrics = gr.HTML(
                "<div style='padding: 20px; text-align: center; color: #999;'>Click 'Run Evaluation' to start</div>"
            )
        with gr.Column(scale=1):
            answer_chart = gr.BarPlot(
                x="Category", y="Average Accuracy", title="Average Accuracy by Category", y_lim=[1, 5], height=450, x_label_angle=-40
            )

    retrieval_button.click(fn=run_retrieval_evaluation, inputs=[test_set, embedding_model, chat_model], outputs=[retrieval_metrics, retrieval_chart], concurrency_limit=2)
    answer_button.click(fn=run_answer_evaluation, inputs=[test_set, embedding_model, chat_model], outputs=[answer_metrics, answer_chart], concurrency_limit=2)


def main():
    """Launch the evaluation dashboard as its own standalone app."""
    theme = gr.themes.Soft(font=["Inter", "system-ui", "sans-serif"])
    with gr.Blocks(title="RAG Evaluation Dashboard", theme=theme) as app:
        build_eval_tab()
    app.launch(inbrowser=True)


if __name__ == "__main__":
    main()
