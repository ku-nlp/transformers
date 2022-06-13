import argparse
import json
from typing import List, Set
import sys
import unicodedata
# from concurrent import futures
from multiprocessing import Pool
from functools import partial

from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizer


def prepare_ref(line: str, tokenizer: PreTrainedTokenizer) -> List[int]:
    encoding = tokenizer(line, add_special_tokens=True, truncation=True, max_length=512)
    input_tokens = []
    ref_ids = []
    ch_idx = 0
    is_word_starts = _get_is_word_starts(tokenizer.word_tokenizer.tokenize(unicodedata.normalize("NFKC", line)))
    tokens = tokenizer.convert_ids_to_tokens(encoding['input_ids'])
    for i, token in enumerate(tokens):
        if token in tokenizer.all_special_tokens:
            continue
        if is_word_starts[ch_idx] is False:
            input_tokens.append("##" + token)
            ref_ids.append(i)
        else:
            input_tokens.append(token)
        ch_idx += len(token)
    # assert ch_idx == len(is_word_starts)
    # print(input_tokens, file=sys.stderr)
    return ref_ids


def _get_is_word_starts(words: List[str]) -> List[bool]:
    is_word_starts = [False] * sum(len(word) for word in words)
    cum_lens = 0
    for word in words:
        is_word_starts[cum_lens] = True
        cum_lens += len(word)
    return is_word_starts


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.bert)

    lines = []
    with open(args.file_name, mode="r") as f:
        for line in tqdm(f):
            if len(line) > 0 and not line.isspace():
                lines.append(line.strip())
    print(f"{len(lines)} lines loaded.", file=sys.stderr)

    # with futures.ProcessPoolExecutor(max_workers=args.n_jobs) as executor:
    #     print(f"{args.n_jobs} workers are running.", file=sys.stderr)
    #     rets: List[List[int]] = executor.map(prepare_ref, lines, [tokenizer] * len(lines))
    output_lines = []
    new_prepare_ref = partial(prepare_ref, tokenizer=tokenizer)
    with Pool(args.n_jobs) as pool:
        print(f"{args.n_jobs} workers are running.", file=sys.stderr)
        for ret in tqdm(pool.imap(new_prepare_ref, lines, chunksize=100000), total=len(lines)):
            output_lines.append(json.dumps(ret) + "\n")
        # rets: List[List[int]] = list(tqdm(pool.imap(new_prepare_ref, lines, chunksize=100000)))
    # print(f"{len(rets)} lines prepared.", file=sys.stderr)

    with open(args.save_path, mode="w") as f:
        f.write("".join(output_lines))
        # for ret in tqdm(rets):
        #     f.write(json.dumps(ret) + "\n")

    #     data = f.readlines()
    # data = [line.strip() for line in data if len(line) > 0 and not line.isspace()]  # avoid delimiter like '\u2029'
    # # ltp_tokenizer = LTP(args.ltp)  # faster in GPU device

    # ref_ids: list = prepare_ref(data, tokenizer)

    # with open(args.save_path, mode="w") as f:
    #     data = [json.dumps(ref) + "\n" for ref in ref_ids]
    #     f.writelines(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="prepare_japanese_ref")
    parser.add_argument(
        "--file-name",
        type=str,
        default="./resources/chinese-demo.txt",
        help="file need process, same as training data in lm",
    )
    parser.add_argument("--bert", type=str, default="./resources/robert", help="resources for Bert tokenizer")
    parser.add_argument("--save-path", type=str, default="./resources/ref.txt", help="path to save res")
    parser.add_argument("--n-jobs", type=int, default=0, help="number of jobs")

    main(parser.parse_args())
