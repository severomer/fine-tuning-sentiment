This is a base fine-tuning example for sentiment analysis.

Used python 3.8

You can start creating a virtual environment:


# 1. Create a virtual environment
python3 -m venv hf_cpu

# 2. Activate it
source hf_cpu/bin/activate   # Linux / macOS
# OR
hf_cpu\Scripts\activate      # Windows PowerShell

# 3. Upgrade pip
pip install --upgrade pip

# 4. Install required packages
pip install torch==2.2.2 transformers==4.44.2 datasets==3.0.1


if you want you can install requirements.txt or step 4.

Then run as:

python fine_tune.py

I run this 1 epoch fine tuning on 8 core server with ✅ CPU training time: 212.68 seconds

Here is the expected output (my output)

Device: cpu
/hf_cpu/lib/python3.8/site-packages/transformers/tokenization_utils_base.py:1601: FutureWarning: `clean_up_tokenization_spaces` was not set. It will be set to `True` by default. This behavior will be depracted in transformers v4.45, and will be then set to `False` by default. For more details check this issue: https://github.com/huggingface/transformers/issues/31884
  warnings.warn(
Some weights of DistilBertForSequenceClassification were not initialized from the model checkpoint at distilbert-base-uncased and are newly initialized: ['classifier.bias', 'classifier.weight', 'pre_classifier.bias', 'pre_classifier.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
/hf_cpu/lib/python3.8/site-packages/transformers/training_args.py:1525: FutureWarning: `evaluation_strategy` is deprecated and will be removed in version 4.46 of 🤗 Transformers. Use `eval_strategy` instead
  warnings.warn(
Detected kernel version 5.4.0, which is below the recommended minimum of 5.5.0; this can cause the process to hang. It is recommended to upgrade the kernel to the minimum version or higher.
Starting fine-tuning...
{'loss': 0.6983, 'grad_norm': 1.372401475906372, 'learning_rate': 4.600000000000001e-05, 'epoch': 0.08}                                       
{'loss': 0.5655, 'grad_norm': 4.981560707092285, 'learning_rate': 4.2e-05, 'epoch': 0.16}                                                     
{'loss': 0.4247, 'grad_norm': 11.860206604003906, 'learning_rate': 3.8e-05, 'epoch': 0.24}                                                    
{'loss': 0.4432, 'grad_norm': 10.714362144470215, 'learning_rate': 3.4000000000000007e-05, 'epoch': 0.32}                                     
{'loss': 0.3727, 'grad_norm': 8.214064598083496, 'learning_rate': 3e-05, 'epoch': 0.4}                                                        
{'loss': 0.507, 'grad_norm': 27.141321182250977, 'learning_rate': 2.6000000000000002e-05, 'epoch': 0.48}                                      
{'loss': 0.4156, 'grad_norm': 1.469207763671875, 'learning_rate': 2.2000000000000003e-05, 'epoch': 0.56}                                      
{'loss': 0.4881, 'grad_norm': 14.21233081817627, 'learning_rate': 1.8e-05, 'epoch': 0.64}                                                     
{'loss': 0.4396, 'grad_norm': 11.08486557006836, 'learning_rate': 1.4000000000000001e-05, 'epoch': 0.72}                                      
{'loss': 0.4128, 'grad_norm': 6.321080684661865, 'learning_rate': 1e-05, 'epoch': 0.8}                                                        
{'loss': 0.3842, 'grad_norm': 7.407820224761963, 'learning_rate': 6e-06, 'epoch': 0.88}                                                       
{'loss': 0.3238, 'grad_norm': 9.31689739227295, 'learning_rate': 2.0000000000000003e-06, 'epoch': 0.96}                                       
{'eval_loss': 0.42040640115737915, 'eval_runtime': 16.8758, 'eval_samples_per_second': 59.256, 'eval_steps_per_second': 7.407, 'epoch': 1.0}  
{'train_runtime': 212.041, 'train_samples_per_second': 9.432, 'train_steps_per_second': 1.179, 'train_loss': 0.45552169609069826, 'epoch': 1.0}                                                                                                                                             
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 250/250 [03:32<00:00,  1.18it/s]
✅ CPU training time: 212.68 seconds

Evaluating fine-tuned model...
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 125/125 [00:17<00:00,  7.32it/s]
Evaluation results: {'eval_loss': 0.42040640115737915, 'eval_runtime': 17.1975, 'eval_samples_per_second': 58.148, 'eval_steps_per_second': 7.268, 'epoch': 1.0}

Testing inference with fine-tuned model:
/hf_cpu/lib/python3.8/site-packages/transformers/tokenization_utils_base.py:1601: FutureWarning: `clean_up_tokenization_spaces` was not set. It will be set to `True` by default. This behavior will be depracted in transformers v4.45, and will be then set to `False` by default. For more details check this issue: https://github.com/huggingface/transformers/issues/31884
  warnings.warn(
[{'label': 'LABEL_1', 'score': 0.9658324122428894}]
[{'label': 'LABEL_0', 'score': 0.9375929236412048}]

Comparing with base (non-finetuned) model:
Some weights of DistilBertForSequenceClassification were not initialized from the model checkpoint at distilbert-base-uncased and are newly initialized: ['classifier.bias', 'classifier.weight', 'pre_classifier.bias', 'pre_classifier.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
[{'label': 'LABEL_1', 'score': 0.5297579765319824}]


