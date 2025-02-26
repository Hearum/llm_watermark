
from evaluate import load
import pdb
perplexity = load("perplexity", module_type="metric")

input_texts = ["cal test "]

results = perplexity.compute(model_id='gpt2',
                             add_start_token=False,
                             predictions=input_texts)
print(results)
pdb.set_trace()