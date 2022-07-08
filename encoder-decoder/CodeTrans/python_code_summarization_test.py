# taken from https://github.com/agemagician/CodeTrans
import torch
from transformers import AutoTokenizer, AutoModelWithLMHead, SummarizationPipeline
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from tree_sitter import Language, Parser

from likelihood_compute_utils import greedy_search, compute_log_likelihood
from parser import my_traverse

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

tree = parser.parse(bytes(code, "utf8"))
code_list = []
tokenized_code = my_traverse(tree.root_node, code, code_list)
print("Output after tokenization: " + tokenized_code)

code_tokens = pipeline.tokenizer.tokenize(tokenized_code)

code_ids = pipeline.tokenizer.convert_tokens_to_ids(code_tokens)

embedding_out = embedding(torch.tensor([code_ids], device=torch.device("cpu")))
print(embedding_out)

encoder_out: BaseModelOutputWithPastAndCrossAttentions = encoder(torch.tensor([code_ids], device=torch.device("cpu")))
print(encoder_out)
#
pipeline_output = greedy_search(
    pipeline.model,
    torch.tensor([[0]]),
    max_length=20,
    attention_mask=torch.ones((1, 41)),
    encoder_outputs=encoder_out
)
print(pipeline_output)
input_ids, sequence_log_likelihood, sequence_likelihood = pipeline_output

# decoder_input_ids = pipeline.tokenizer('<pad>', return_tensors='pt').input_ids
#
# decoder_out = decoder(decoder_input_ids, encoder_hidden_states=encoder_out.last_hidden_state)
# print(decoder_out)
#
# t5_out = output(decoder_out.last_hidden_state)
#
# result = pipeline.tokenizer.convert_ids_to_tokens(t5_out.argmax(axis=2).tolist()[0])
# print(result)

text_out = pipeline.tokenizer.decode(input_ids.tolist()[0])
print(text_out)
print("log-likelihood", sequence_log_likelihood)
print("likelihood", sequence_likelihood)

### <pad> Prints an error message and exits with a default exit code.</s>
### log-likelihood 233.66766834259033
### likelihood 6.624599110151735e+17


log_likelihood = compute_log_likelihood(
    pipeline.model,
    input_ids,
    max_length=20,
    attention_mask=torch.ones((1, 41)),
    encoder_outputs=encoder_out
)

print("log-likelihood-2", log_likelihood)

### log-likelihood-2 tensor(233.6677, grad_fn=<AddBackward0>)
