# taken from https://github.com/agemagician/CodeTrans
import torch
from transformers import AutoTokenizer, AutoModelWithLMHead, SummarizationPipeline
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from tree_sitter import Language, Parser

pipeline = SummarizationPipeline(
    model=AutoModelWithLMHead.from_pretrained("SEBIS/code_trans_t5_small_code_documentation_generation_python"),
    tokenizer=AutoTokenizer.from_pretrained("SEBIS/code_trans_t5_small_code_documentation_generation_python",
                                            skip_special_tokens=True),
    device=torch.device("cpu")
)

embedding, encoder, decoder, output = list(pipeline.model.children())

code = "def e(message, exit_code=None):\n   print_log(message, YELLOW, BOLD)\n" \
       "    if exit_code is not None:\n        sys.exit(exit_code)"  # @param {type:"raw"}

Language.build_library(
    'build/my-languages.so',
    ['tree-sitter-python']
)

PYTHON_LANGUAGE = Language('build/my-languages.so', 'python')
parser = Parser()
parser.set_language(PYTHON_LANGUAGE)


def get_string_from_code(node, lines):
    line_start = node.start_point[0]
    line_end = node.end_point[0]
    char_start = node.start_point[1]
    char_end = node.end_point[1]
    if line_start != line_end:
        code_list.append(
            ' '.join([lines[line_start][char_start:]] + lines[line_start + 1:line_end] + [lines[line_end][:char_end]]))
    else:
        code_list.append(lines[line_start][char_start:char_end])


def my_traverse(node, code_list):
    lines = code.split('\n')
    if node.child_count == 0:
        get_string_from_code(node, lines)
    elif node.type == 'string':
        get_string_from_code(node, lines)
    else:
        for n in node.children:
            my_traverse(n, code_list)

    return ' '.join(code_list)


tree = parser.parse(bytes(code, "utf8"))
code_list = []
tokenized_code = my_traverse(tree.root_node, code_list)
print("Output after tokenization: " + tokenized_code)

code_tokens = pipeline.tokenizer.tokenize(tokenized_code)
code_ids = pipeline.tokenizer.convert_tokens_to_ids(code_tokens)

embedding_out = embedding(torch.tensor([code_ids], device=torch.device("cpu")))
print(embedding_out)

encoder_out: BaseModelOutputWithPastAndCrossAttentions = encoder(torch.tensor([code_ids], device=torch.device("cpu")))
print(encoder_out)

decoder_input_ids = pipeline.tokenizer('<pad>', return_tensors='pt').input_ids

decoder_out = decoder(decoder_input_ids, encoder_hidden_states=encoder_out.last_hidden_state)
print(decoder_out)

t5_out = output(decoder_out.last_hidden_state)

result = pipeline.tokenizer.convert_ids_to_tokens(t5_out.argmax(axis=2).tolist()[0])
print(result)
