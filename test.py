import torch
from utils_1 import *
from models.encoders.DFormerv2 import DFormerv2_L  # 或 DFormerv2_S / DFormerv2_L

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. 创建模型实例
model = DFormerv2_L()  # 也可以选择 DFormerv2_S、DFormerv2_L
model.to(device)

# 2. 读取权重文件
ckpt_path = "checkpoints/DFormerv2_Large_pretrained.pth"
state_dict = torch.load(ckpt_path, map_location=device)

# 3. 兼容不同保存格式（如果权重里包了 state_dict）
if "state_dict" in state_dict:
    state_dict = state_dict["state_dict"]

# for i in state_dict:
#     # if 'backbone' and 's.0' in i:
#     print(i)
# 4. 加载参数
model.load_state_dict(state_dict, strict=True)

# 5. 切换到推理模式（可选）
model.eval()

train_set = ISPRS_dataset(train_ids, cache=CACHE)
train_loader = torch.utils.data.DataLoader(train_set,batch_size=BATCH_SIZE)
for batch_idx, (data, dsm, target) in enumerate(train_loader):
    data, dsm, target = data.cuda(), dsm.cuda(), target.cuda()
    f4, f8, f16, f32 = model(data, dsm)
