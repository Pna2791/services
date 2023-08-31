import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from datetime import datetime
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
# cors = CORS(app, resources={r"/*": {"origins": "*"}})


class Generator:
    def __init__(self, model_name='vilm/vietcuna-3b') -> None:
        print(f"Starting to load the model {model_name} into memory")

        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True
        )
        self.model_raw = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
            device_map={"": 0}
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, padding_side='left')
        print(f"Successfully loaded the model {model_name} into memory")

    def generate(self, prompt):
        t_start = time.time()
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        input_ids = input_ids.to(self.model_raw.device)

        outputs = self.model_raw.generate(
            input_ids=input_ids, max_new_tokens=256, min_new_tokens=True, early_stopping=True)
        answer = self.tokenizer.decode(
            outputs.cpu()[0], skip_special_tokens=True)
        print(len(outputs.cpu()[0]))

        p_len = len(prompt)
        return answer[p_len+1:], round(time.time() - t_start, 2)


generator = Generator()


def show_time():
    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
    return f"[{formatted_time}]"


@app.route('/chat', methods=['POST'])
def process_data():
    data = request.get_json()
    print(show_time(), "Data:", data)
    prompt = data['prompt']

    # You need to define 'generator' somewhere
    result, duration = generator.generate(prompt)
    output = {'status': 'done', 'result': result, 'duration': duration}
    return jsonify(output)


@app.route("/")
def home():
    return "<h1>GFG is a great platform to learn</h1>"


if __name__ == "__main__":
    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")

    print(formatted_time)
    app.run(host="0.0.0.0")
