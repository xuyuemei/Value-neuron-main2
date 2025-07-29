import torch
import torch.nn.functional as F
from collections import defaultdict
import numpy as np
from sklearn.feature_selection import mutual_info_classif


def activation(value, over_zero, token_num, top_rate, filter_rate, activation_bar_ratio, num_layers):
    top_rate = top_rate
    filter_rate = filter_rate
    activation_bar_ratio = activation_bar_ratio
    activation_probs = over_zero / token_num  # layer x inter x lang_num
    # print(normed_weight[0][0])
    normed_activation_probs = activation_probs / activation_probs.sum(dim=-1, keepdim=True)
    normed_activation_probs[torch.isnan(normed_activation_probs)] = 0
    # print(normed_activation_probs)
    log_probs = torch.where(normed_activation_probs > 0, normed_activation_probs.log(), 0)
    entropy = -torch.sum(normed_activation_probs * log_probs, dim=-1)
    # print(entropy)
    largest = False

    if torch.isnan(entropy).sum():
        print(torch.isnan(entropy).sum())
        raise ValueError

    flattened_probs = activation_probs.flatten()
    top_prob_value = flattened_probs.kthvalue(round(len(flattened_probs) * filter_rate)).values.item()
    print("top_prob_value", top_prob_value)
    # dismiss the neruon if no language has an activation value over top 90%
    top_position = (activation_probs > top_prob_value).sum(dim=-1)
    entropy[top_position == 0] = -torch.inf if largest else torch.inf

    flattened_entropy = entropy.flatten()
    sorted_entropy, sorted_indices = torch.sort(flattened_entropy)
    print("Sorted Entropy:", sorted_entropy)
    top_entropy_value = round(len(flattened_entropy) * top_rate)
    print("Top_entropy_value:", sorted_entropy[top_entropy_value])
    _, index = flattened_entropy.topk(top_entropy_value, largest=largest)
    row_index = index // entropy.size(1)
    col_index = index % entropy.size(1)
    selected_probs = activation_probs[row_index, col_index]  # n x lang
    # for r, c in zip(row_index, col_index):
    #     print(r, c, activation_probs[r][c])

    print(selected_probs.size(0), torch.bincount(selected_probs.argmax(dim=-1)))
    selected_probs = selected_probs.transpose(0, 1)
    activation_bar = flattened_probs.kthvalue(round(len(flattened_probs) * activation_bar_ratio)).values.item()
    print((selected_probs > activation_bar).sum(dim=1).tolist())
    lang, indice = torch.where(selected_probs > activation_bar)
    print(len(lang), len(indice))

    combined = torch.stack([indice, lang], dim=1)
    # 创建 defaultdict 以存储 lang 相同的 indice
    grouped = defaultdict(list)

    # 遍历 combined，将 indice 添加到对应的 lang 列表中
    for ind, l in combined.tolist():
        grouped[l].append(ind)


    # 将结果转为标准字典（可选）
    grouped = dict(grouped)
    # print(grouped)

    # 获取唯一的 lang 值
    unique_langs = torch.unique(lang)

    # 初始化 overlap 矩阵
    overlap_matrix = torch.zeros((len(unique_langs), len(unique_langs)), dtype=torch.int)

    # 统计每个 lang 对应的 indices 集合

    # 计算每两种语言的 overlap
    for i, lang1 in enumerate(unique_langs):
        for j, lang2 in enumerate(unique_langs):
            # 取两个语言对应的 indices 集合
            indices1 = set(grouped[lang1.item()])
            indices2 = set(grouped[lang2.item()])
            # 计算交集的大小
            overlap = len(indices1.intersection(indices2))
            overlap_matrix[i, j] = overlap
            if i ==j:
                overlap_matrix[i, j] = 0


    # 输出结果
    print("Overlap Matrix:")
    print(overlap_matrix)


    merged_index = torch.stack((row_index, col_index), dim=-1)
    final_indice = []
    entropy_record = []
    activation_record = []
    for _, index in enumerate(indice.split(torch.bincount(lang).tolist())):
        lang_index = [tuple(row.tolist()) for row in merged_index[index]]
        lang_index.sort()
        layer_index = [[] for _ in range(num_layers)]
        entropy_layer = [[] for _ in range(num_layers)]
        activation_layer = [[] for _ in range(num_layers)]
        for l, h in lang_index:
            layer_index[l].append(h)
            entropy_layer[l].append(entropy[l][h])
            activation_layer[l].append(normed_activation_probs[l][h])
        for l, h in enumerate(layer_index):
            layer_index[l] = torch.tensor(h).long()
            entropy_layer[l] = torch.tensor(entropy[l][h])
            activation_layer[l] = torch.tensor(normed_activation_probs[l][h])
        final_indice.append(layer_index)
        entropy_record.append(entropy_layer)
        activation_record.append(activation_layer)
    if filter_rate == 0.5:
        neuron_type = 'semantic'
    else:
        neuron_type = 'decision'
    torch.save(final_indice, f"/!!nips/value_neuron-Qwen/activation_mask_{value}_{neuron_type}_{top_rate}")
    torch.save(entropy_record, f"/!!nips/record-Qwen/entropy_{value}_{neuron_type}_{top_rate}")
    torch.save(activation_record, f"/!!nips/record-Qwen/activation_{value}_{neuron_type}_{top_rate}")


value_list = ["富强","民主","文明","和谐","自由","平等","公正","法治","爱国","敬业","诚信","友善"]

for value in value_list:
    token_num, question_num, over_zero = [], [], []
    for condition in [f"{value}-support", f"{value}-conflict"]:
        data = torch.load(f'/!!nips/Qwen-neuron/{condition}.pt')
        token_num.append(int(data['token_num']))
        question_num.append(data['question_num'])
        over_zero.append(data['over_zero'])

    token_num = torch.tensor(token_num)
    print("*******",token_num)
    token_num = torch.tensor(question_num)
    print("*******",question_num)
    over_zero = torch.stack(over_zero, dim=-1)
    print(over_zero.shape)

    num_layers, _, _ = over_zero.size()

    activation(value, over_zero, token_num, 0.01, 0.50, 0.50, num_layers)
    activation(value, over_zero, token_num, 0.01, 0.95, 0.95, num_layers)







