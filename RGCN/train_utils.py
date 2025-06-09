import logging
import torch
from tqdm import tqdm
from torch_geometric.utils import negative_sampling
from data_utils import create_inference_loader


def train(model, train_loader, optimizer, device='cuda', num_negatives=7, hard_k=2):
    model.train()
    total_loss = 0
    batch_idx = 0
    for batch in tqdm(train_loader, desc='Training'):
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x_dict, batch.edge_index_dict)
         # Compute BPR loss over all user–item relations
        loss_batch = 0
        rel_count = 0
        for edge_type in batch.edge_index_dict:
            if edge_type[0] != 'user' or edge_type[-1] != 'item':
                continue
            edge_index = batch[edge_type].edge_index
            
            if edge_index.size(1) == 0:
                continue

            u_emb = out[edge_type[0]]  
            i_emb = out[edge_type[-1]]  
            
            # 1. Vectorized sampling of negatives per positive edge
            num_items = i_emb.size(0)
            u_idx = edge_index[0]                    # [N_pos]
            v_pos = edge_index[1]                   # [N_pos]

            # --- only keep positives whose user is a seed node ---
            # NeighborLoader places seed nodes first, up to batch_size
            seed_batch_size = train_loader.batch_size
            seed_mask = u_idx < seed_batch_size
            if seed_mask.sum() == 0:
                continue
            u_idx = u_idx[seed_mask]
            v_pos = v_pos[seed_mask]
            
            # Sample `num_negatives` random negatives per positive
            neg_v = torch.randint(
                0, num_items,
                (u_idx.size(0), num_negatives),
                device=device
            )                                       # [N_pos, num_negatives]

            # 2. Compute scores for all negatives
            neg_scores_mat = (
                u_emb[u_idx].unsqueeze(1) *    # [N_pos, 1, D]
                i_emb[neg_v]                   # [N_pos, num_negatives, D]
            ).sum(dim=2)                        # [N_pos, num_negatives]

            # 3. Take top-k hardest per positive
            k = min(hard_k, num_negatives)
            _, topk_idx = neg_scores_mat.topk(k, dim=1)        # [N_pos, k]
            v_hard_neg   = torch.gather(neg_v, 1, topk_idx)    # [N_pos, k]

            # + two extra random negatives via pyg negative_sampling
            rand_k = 2
            neg_edge = negative_sampling(
                edge_index,
                num_nodes=num_items,
                num_neg_samples=u_idx.size(0) * rand_k
            )  # [2, N_pos*rand_k]
            v_rand_neg = neg_edge[1].view(u_idx.size(0), rand_k)  # [N_pos, rand_k]

            # combine hard + random negatives
            v_all_neg = torch.cat([v_hard_neg, v_rand_neg], dim=1)  # [N_pos, k+rand_k]
            k_all = v_all_neg.size(1)

            # 4. Build final u / pos / neg arrays
            final_u   = u_idx.unsqueeze(1).expand(-1, k_all).reshape(-1)
            final_pos = v_pos.unsqueeze(1).expand(-1, k_all).reshape(-1)
            final_neg = v_all_neg.reshape(-1)

            # Remove any cases where the "negative" equals the positive:
            mask = final_neg != final_pos
            final_u   = final_u[mask]
            final_pos = final_pos[mask]
            final_neg = final_neg[mask]
            
            # 5. Lookup embeddings and compute BPR
            u_final_emb   = u_emb[final_u]
            pos_final_emb = i_emb[final_pos]
            neg_final_emb = i_emb[final_neg]

            pos_scores = (u_final_emb * pos_final_emb).sum(dim=1)
            neg_scores = (u_final_emb * neg_final_emb).sum(dim=1)

            diff = pos_scores - neg_scores

            bpr  = -torch.log(torch.sigmoid(diff) + 1e-15).mean()

            if loss_batch == 0:
                loss_batch = bpr
            else:
                loss_batch += bpr
            rel_count += 1

        loss_batch.backward()
        optimizer.step()
        total_loss += loss_batch.item()
        batch_idx += 1
        if batch_idx % 1000 == 0:
            logging.info(f"Batch {batch_idx:03d} loss={loss_batch:.4f}")
    return total_loss / len(train_loader)


def extract_user_embeddings(model, data, device, batch_size, num_workers, out_channels, output_path):
    model.eval()
    inference_loader = create_inference_loader(data, batch_size, num_workers)
    num_users = data['user'].num_nodes
    all_user_emb = torch.zeros(num_users, out_channels)
    for batch_idx, batch in enumerate(tqdm(inference_loader, desc="Inferring user embeddings")):
        batch = batch.to(device)
        out = model(batch.x_dict, batch.edge_index_dict)
        user_idx = batch['user'].n_id.cpu()
        emb = out['user'].cpu()
        all_user_emb[user_idx] = emb
    torch.save(all_user_emb, output_path)
    logging.info(f"Saved user embeddings: {output_path}")
