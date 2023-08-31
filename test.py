from datetime import datetime
import torch
import time
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

context_list = torch.load(
    '/media/creator/ac9a5c2e-d7d9-41db-93a7-1b3d2b8ac013/Script/my_dataset/context.pt')


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
            # model_name,
            # load_in_8bit=True,
            # torch_dtype=torch.bfloat16,
            # device_map={"": 0}
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, padding_side='left')
        print(f"Successfully loaded the model {model_name} into memory")

    def generate(self, prompt):
        t_start = time.time()
        input_ids = self.tokenizer(
            prompt, padding=True, truncation=True, return_tensors='pt').input_ids
        input_ids = input_ids.to(self.model_raw.device)

        outputs = self.model_raw.generate(
            input_ids=input_ids, max_new_tokens=256, min_new_tokens=True, early_stopping=True)
        answer = self.tokenizer.decode(
            outputs.cpu()[0], skip_special_tokens=True)
        print(len(outputs.cpu()[0]))

        p_len = len(prompt[0])
        return answer[p_len+1:], round(time.time() - t_start, 2)


generator = Generator()


context = """Chặng đường phát triển
Năm 1989, thành lập Công ty Điện tử thiết bị thông tin, là tiền thân của Tổng Công ty Viễn thông Quân đội (Viettel)[2][3].

Năm 1995, đổi tên Công ty Điện tử thiết bị thông tin thành Công ty Điện tử Viễn thông Quân đội[4] (tên giao dịch là Viettel) chính thức trở thành nhà cung cấp dịch vụ viễn thông thứ hai tại Việt Nam.[2]

Năm 2000, Viettel được cấp giấy phép cung cấp thử nghiệm dịch vụ điện thoại đường dài sử dụng công nghệ VoIP tuyến Hà Nội – Hồ Chí Minh với thương hiệu 178 và đã triển khai thành công.[2]

Năm 2003, Viettel bắt đầu đầu tư vào những dịch vụ viễn thông cơ bản, lắp đặt tổng đài đưa dịch vụ điện thoại cố định vào hoạt động kinh doanh trên thị trường.[3] Viettel cũng thực hiện phổ cập điện thoại cố định tới tất cả các vùng miền trong cả nước với chất lượng phục vụ ngày càng cao.[4]

Ngày 15 tháng 10 năm 2004, mạng di động 098 chính thức đi vào hoạt động đánh dấu một bước ngoặt trong sự phát triển của Viettel Mobile và Viettel.[3]

Ngày 2 tháng 3, năm 2005, Tổng Công ty Viễn thông quân đội theo quyết định của Thủ tướng Phan Văn Khải và ngày 6 tháng 4 năm 2004, theo quyết định 45/2005/BQP của Bộ Quốc phòng Việt Nam[4] thành lập Tổng Công ty Viễn thông quân đội[3].

Ngày 5 tháng 4 năm 2007, Công ty Viễn thông Viettel (Viettel Telecom) trực thuộc Tổng Công ty Viễn thông Quân đội Viettel được thành lập, trên cơ sở sáp nhập các Công ty Internet Viettel, Điện thoại cố định Viettel và Điện thoại di động Viettel[2].
"""

messages = "Viettel thành lập năm bao nhiêu? Nếu không có thông tin thì hãy trả lời Tôi không biết"
# messages = "Viettel Telecom thành lập năm bao nhiêu?"
prompt = f"Dựa vào thông tin: ```{context}```\n Trả lời câu hỏi sau: ```{messages}```\n Câu trả lời của bạn là:"


for i in range(6):
    batch_size = 2**i
    print('Generating batch size of:', batch_size)
    prompts = []
    for k in range(batch_size):
        context = context_list[k]
        prompts.append(
            f"Dựa vào thông tin: ```{context}```\n Trả lời câu hỏi sau: ```{messages}```\n Câu trả lời của bạn là:")

    print(generator.generate(prompts))
    print('Generating batch size of:', batch_size, 'finished')
