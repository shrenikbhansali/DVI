"""Generate answers with DVI speculative decoding."""
import argparse
import os
import torch

from fastchat.utils import str_to_torch_dtype
from evaluation.eval import run_eval, reorg_answer_file
from transformers import AutoTokenizer
from kangaroo.kangaroo_model import KangarooModel
from training.spec_decode import generate_with_dvi_spec


def kangaroo_forward(inputs, model, tokenizer, max_new_tokens, do_sample=False,
                     EARLY_STOP_LAYER=2, SPECULATIVE_DECODING_STEPS=6, **kwargs):
    enc = {"input_ids": inputs.input_ids}
    outputs, metrics = generate_with_dvi_spec(
        model,
        tokenizer,
        enc=enc,
        max_new_tokens=max_new_tokens,
        draft_k=SPECULATIVE_DECODING_STEPS,
        greedy=not do_sample,
        early_layer=EARLY_STOP_LAYER,
        temperature=kwargs.get("temperature", 1.0),
    )
    seq = torch.cat([inputs.input_ids[0], outputs[0]], dim=0).tolist()
    new_token = len(outputs[0])
    idx = metrics.steps
    acc_list = [metrics.accepted // max(1, metrics.steps)] * max(1, metrics.steps)
    return [seq], new_token, idx, acc_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--adapter-path", type=str, required=True)
    parser.add_argument("--model-id", type=str, required=True)
    parser.add_argument("--bench-name", type=str, default="mt_bench")
    parser.add_argument("--question-begin", type=int)
    parser.add_argument("--question-end", type=int)
    parser.add_argument("--answer-file", type=str)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--num-choices", type=int, default=1)
    parser.add_argument("--num-gpus-per-model", type=int, default=1)
    parser.add_argument("--num-gpus-total", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--exitlayer", type=int, default=2)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--dtype", type=str, default="float16",
                        choices=["float32", "float64", "float16", "bfloat16"],
                        help="Override the default dtype. If not set, it will use float16 on GPU.")
    args = parser.parse_args()

    question_file = "data/question.jsonl"
    model = KangarooModel(args.model_path, args.adapter_path, args, EARLY_STOP_LAYER=args.exitlayer)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    do_sample = False

    assert not args.answer_file
    os.makedirs(f"data/{args.bench_name}/{args.model_id}", exist_ok=True)

    for run in range(3):
        answer_file = f"data/{args.bench_name}/{args.model_id}/{run}.jsonl"
        print(f"Output to {answer_file}")
        run_eval(
            model=model,
            tokenizer=tokenizer,
            forward_func=kangaroo_forward,
            model_id=args.model_id,
            question_file=question_file,
            question_begin=args.question_begin,
            question_end=args.question_end,
            answer_file=answer_file,
            max_new_tokens=args.max_new_tokens,
            num_choices=args.num_choices,
            num_gpus_per_model=args.num_gpus_per_model,
            num_gpus_total=args.num_gpus_total,
            do_sample=do_sample,
            threshold=args.threshold,
            SPECULATIVE_DECODING_STEPS=args.steps,
            EARLY_STOP_LAYER=args.exitlayer,
        )
        reorg_answer_file(answer_file)
