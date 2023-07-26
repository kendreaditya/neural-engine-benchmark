from transformers import AutoTokenizer, AutoModelForCausalLM

checkpoint = "bigscience/bloomz-1b1"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint)

tokenizer.save_pretrained("local-pt-checkpoint")
model.save_pretrained("local-pt-checkpoint")
