from transformers import TensorRTModel, AutoTokenizer
import torch

model_path = "model/topic-11/run-0binary"
trt_engine_dir = "./trt_engines"
max_batch_size = 1024
max_seq_length = 256

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = TensorRTModel.from_pretrained(
    model_path,
    fp16=True,
    max_batch_size=max_batch_size,
    max_workspace_size=16384,
    output_dir=trt_engine_dir,
)

dummy_input = tokenizer(
    ["这是一个测试文本"], 
    max_length=max_seq_length, 
    padding="max_length", 
    return_tensors="pt"
).to("cuda:0")

# 触发引擎构建（耗时几分钟）
with torch.inference_mode():
    outputs = model(**dummy_input)
print("TensorRT引擎构建完成！保存至:", trt_engine_dir)