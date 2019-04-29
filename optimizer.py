import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as I
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable

from tqdm import tqdm
import numpy as np

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.lin1 = nn.Linear(1,1, bias=False)
        self.lin2 = nn.Linear(1,1, bias=False)
        self.lin3 = nn.Linear(1,1, bias=False)
        self.lin4 = nn.Linear(1,1, bias=False)
        self.lin5 = nn.Linear(1,1, bias=False)

        self.lin6 = nn.Linear(1,1, bias=False)
        self.lin7 = nn.Linear(1,1, bias=False)
        self.lin8 = nn.Linear(1,1, bias=False)
        self.lin9 = nn.Linear(1,1, bias=False)
        self.lin10 = nn.Linear(1,1, bias=False)

        self.lin11 = nn.Linear(1,1, bias=False)
        self.lin12 = nn.Linear(1,1, bias=False)

    def forward(self, x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12):
        x1_out = self.lin1(x_1)
        x2_out = self.lin2(x_2)
        x3_out = self.lin3(x_3)
        x4_out = self.lin4(x_4)
        x5_out = self.lin5(x_5)

        x6_out = self.lin6(x_6)
        x7_out = self.lin7(x_7)
        x8_out = self.lin8(x_8)
        x9_out = self.lin9(x_9)
        x10_out = self.lin10(x_10)

        x11_out = self.lin11(x_11)
        x12_out = self.lin12(x_12)

        final_out = x1_out + x2_out + x3_out + x4_out + x5_out
        final_out += x6_out + x7_out + x8_out + x9_out + x10_out
        final_out += x11_out + x12_out

        return final_out

model = Model()

label2id = {"others":0, "happy":1,"sad":2,"angry":3}
id2label = ["others", "happy" ,"sad","angry"]

file_list =[
    "save_final/LIST/test_deepmoji+elmo_0.6987.txt",
    "save_final/LIST/test_fix_hier_pooled_bert_drop0.3_emoji300_lasthidden_voting_prediction.txt",
    "save_final/LIST/test_fix_hier_pooled_bert_drop0.3_super0.3_emoji300.txt",
    "save_final/LIST/test_HLSTM_0.7397.txt",
    "save_final/LIST/test_HLSTM_1000_DROP0.2_ATTN_GLOVE_voting_prediction.txt",
    "save_final/LIST/test_HLSTM_1000_DROP0.2_ATTN_MULTI_HEAD2_GLOVE_SUM_voting_prediction.txt",
    "save_final/LIST/test_HLSTM_1000_DROP0.4_ATTN_GLOVE_EMO2VEC_voting_prediction.txt",
    "save_final/LIST/test_HLSTM_1000_DROP0.4_ATTN_GLOVE_voting_prediction.txt",
    "save_final/LIST/test_HLSTM_1000_DROP0.4_ATTN_MULTI_HEAD2_GLOVE_SUM_voting_prediction.txt",
    "save_final/LIST/test_HLSTM_1500_DROP0.3_ATTN_GLOVE_voting_prediction.txt",
    "save_final/LIST/test_hutrs_emb_488_gp.txt",
    "save_final/LIST/test_hutrs_emb100_voting_prediction.txt"
]

y_list = "save_final/LIST/ensemble27_happy_cut_sad_cut_others3.txt"

def intToVector(num):
    vector = np.zeros(4)
    vector[num] = 1
    return vector.tolist()

data_x = []
for i in range(len(file_list)):
    file = file_list[i]
    data_row = []
    with open(file, "r") as file_in:
        header = True
        for line in file_in:
            if header:
                header=False
                continue
            label = label2id[line.split("\t")[4].replace("\n", "")]
            vector = intToVector(label)
            data_row.append(vector)
    data_x.append(data_row)

data_y = []
i = 0
with open(file, "r") as file_in:
    header = True
    for line in file_in:
        if header:
            header=False
            continue
        label = label2id[line.split("\t")[4].replace("\n", "")]
        data_y.append(label)
        i +=1 

data_x = torch.FloatTensor(data_x).cuda().transpose(0, 1).unsqueeze(-1) # samples, num_pred, 4, 1
data_y = torch.LongTensor(data_y).cuda() # num_samples, 4

model = model.cuda()
criterion = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(),lr=0.001)

best_epoch = 0
best_loss = 10000000000000000000
for i in range(10000):
    batch_size = 5509
    num_batch = data_x.size(0) // batch_size
    total_loss = 0
    print(data_x.size(), data_y.size())

    pbar = tqdm(range(num_batch))
    for j in tqdm(pbar):
        start_idx = j * batch_size
        end_idx = (j+1) * batch_size

        data = data_x[start_idx:end_idx].transpose(0, 1)
        out_vector = model(data[0],data[1],data[2],data[3],data[4],data[5],data[6],data[7],data[8],data[9],data[10],data[11])
        target = data_y[start_idx:end_idx]

        if i == 9999:
            for k in range(out_vector.size(0)):
                print(">>>>>>>", k, out_vector[k]) 
                # print(nn.Softmax(dim=0)(out_vector[k]))

        loss = criterion(out_vector.contiguous().squeeze(), target)
        loss.backward()
        opt.step()

        for p in model.parameters():
            p.data.clamp_(0, 1)

        total_loss += loss.item()
        pbar.set_description("(Epoch {}) IT {}/{} TRAIN LOSS:{:.4f}".format((i),j+1,num_batch, total_loss/(j+1)))

    print("epoch", i, "loss", total_loss/num_batch)
    if model.lin1.weight.data.item() > 0:
        if model.lin2.weight.data.item() > 0:
            if model.lin3.weight.data.item() > 0:
                if model.lin4.weight.data.item() > 0:
                    if model.lin5.weight.data.item() > 0:
                        if model.lin6.weight.data.item() > 0:
                            if model.lin7.weight.data.item() > 0:
                                if model.lin8.weight.data.item() > 0:
                                    if model.lin9.weight.data.item() > 0:
                                        if model.lin10.weight.data.item() > 0:
                                            if model.lin11.weight.data.item() > 0:
                                                if model.lin12.weight.data.item() > 0:
                                                    if best_loss > (total_loss / num_batch):
                                                        print("lin_1", model.lin1.weight.data.item())
                                                        print("lin_2", model.lin2.weight.data.item())
                                                        print("lin_3", model.lin3.weight.data.item())
                                                        print("lin_4", model.lin4.weight.data.item())
                                                        print("lin_5", model.lin5.weight.data.item())
                                                        print("lin_6", model.lin6.weight.data.item())
                                                        print("lin_7", model.lin7.weight.data.item())
                                                        print("lin_8", model.lin8.weight.data.item())
                                                        print("lin_9", model.lin9.weight.data.item())
                                                        print("lin_10", model.lin10.weight.data.item())
                                                        print("lin_11", model.lin11.weight.data.item())
                                                        print("lin_12", model.lin12.weight.data.item())
                                                        lin_1_weight = model.lin1.weight.data.item()
                                                        lin_2_weight = model.lin2.weight.data.item()
                                                        lin_3_weight = model.lin3.weight.data.item()
                                                        lin_4_weight = model.lin4.weight.data.item()
                                                        lin_5_weight = model.lin5.weight.data.item()
                                                        lin_6_weight = model.lin6.weight.data.item()
                                                        lin_7_weight = model.lin7.weight.data.item()
                                                        lin_8_weight = model.lin8.weight.data.item()
                                                        lin_9_weight = model.lin9.weight.data.item()
                                                        lin_10_weight = model.lin10.weight.data.item()
                                                        lin_11_weight = model.lin11.weight.data.item()
                                                        lin_12_weight = model.lin12.weight.data.item()
                                                        best_epoch = i

new_model = Model()
new_model.lin1.data = torch.FloatTensor([lin_1_weight])
new_model.lin2.data = torch.FloatTensor([lin_2_weight])
new_model.lin3.data = torch.FloatTensor([lin_3_weight])
new_model.lin4.data = torch.FloatTensor([lin_4_weight])
new_model.lin5.data = torch.FloatTensor([lin_5_weight])
new_model.lin6.data = torch.FloatTensor([lin_6_weight])
new_model.lin7.data = torch.FloatTensor([lin_7_weight])
new_model.lin8.data = torch.FloatTensor([lin_8_weight])
new_model.lin9.data = torch.FloatTensor([lin_9_weight])
new_model.lin10.data = torch.FloatTensor([lin_10_weight])
new_model.lin11.data = torch.FloatTensor([lin_11_weight])
new_model.lin12.data = torch.FloatTensor([lin_12_weight])

batch_size = 5509
num_batch = data_x.size(0) // batch_size

pbar = tqdm(range(num_batch))
for j in tqdm(pbar):
    start_idx = j * batch_size
    end_idx = (j+1) * batch_size

    data = data_x[start_idx:end_idx].transpose(0, 1)
    out_vector = model(data[0],data[1],data[2],data[3],data[4],data[5],data[6],data[7],data[8],data[9],data[10],data[11])
    target = data_y[start_idx:end_idx]

with open("guess.txt", "w") as file_out:
    file_out.write(str(lin_1_weight) + "\t" + str(lin_2_weight) + "\t" )
    file_out.write(str(lin_3_weight) + "\t" + str(lin_4_weight) + "\t" )
    file_out.write(str(lin_5_weight) + "\t" + str(lin_6_weight) + "\t" )
    file_out.write(str(lin_7_weight) + "\t" + str(lin_8_weight) + "\t" )
    file_out.write(str(lin_9_weight) + "\t" + str(lin_10_weight) + "\t" )
    file_out.write(str(lin_11_weight) + "\t" + str(lin_12_weight) + "\n")
    for k in range(out_vector.size(0)):
        idx = np.argmax(out_vector[k].squeeze().detach().cpu().numpy())
        file_out.write(str(k) + "\t" + id2label[idx] + "\n")
        print(str(k) + "\t" + id2label[idx])


