import torch 

def mixup_data_euc_avg(x, W, labels):
    batch_size = x.size()[0]
    labelmix = torch.zeros(batch_size, dtype=torch.int64) 
    index = []
    w_mix = torch.zeros(W.size(0), batch_size) 
    y_mix = torch.zeros(batch_size, dtype=torch.int64) 
    set_label = list(set(labels.cpu().detach().numpy()))
    weight_vectors = [W[:, speaker] for speaker in set_label]
    dic_spk = {}
    distances = {}
    for single_spk in set_label:
        distances = [torch.dist(W[:, single_spk], W[:,speaker]) for speaker in set_label if single_spk != speaker]
        closest_neighbor_index = torch.argmin(torch.tensor(distances))
        closest_speaker = set_label[closest_neighbor_index]            
        if single_spk == closest_speaker.item():
            sorted_distances, sorted_indices = torch.sort(torch.tensor(distances))
            # Get the second minimum distance and its corresponding speaker
            second_min_distance = sorted_distances[1]
            second_closest_neighbor_index = sorted_indices[1]
            second_closest_speaker = set_label[second_closest_neighbor_index]
            dic_spk[single_spk] = second_closest_speaker.item()
        else:
            dic_spk[single_spk] = closest_speaker.item()
    lst_labels = labels.tolist()
    newlabel = {}
    labelid = 0
    for i in range(batch_size):
        l1 = labels[i].item()
        l2 = dic_spk[l1] 
        dictidx = int(str(int(l1)) + str(int(l2)))
        if dictidx not in newlabel:
            newlabel[dictidx] = labelid
            w_mix[:,labelid] = (W[:, l1] + W[:, l2])/2
            labelid = labelid + 1
        else:
            w_mix[:,newlabel[dictidx]] = (W[:, l1] + W[:, l2])/2
        y_mix[i] = newlabel[dictidx]
        index.append(lst_labels.index(l2))
    x_mix = 0.5*(x + x[index,:])
    
    x_combined = x_mix
    w_combined = w_mix[:, 0:labelid].to(x.device)
    y_combined = y_mix.to(x.device)
    return x_combined, y_combined , w_combined

