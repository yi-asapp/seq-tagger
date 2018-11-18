import torch
import torch.nn as nn
import torch.nn.functional as F


def log_sum_exp(inp, m_size):
    """
    LogSumExp trick: https://en.wikipedia.org/wiki/LogSumExp

    Args:
        inp: size=(batch_size, vanishing_dim, hidden_dim)
        m_size: hidden_dim

    Returns:
        out: size=(batch_size, hidden_dim)
    """
    _, idx = torch.max(inp, 1)  # batch_size * hidden_dim
    max_score = torch.gather(inp, 1, idx.view(-1, 1, m_size)).view(-1, 1, m_size)  # batch_size * 1 * hidden_dim
    return max_score.view(-1, m_size) + torch.log(torch.sum(
        torch.exp(inp - max_score.expand_as(inp)), 1)).view(-1, m_size)


class CRF(nn.Module):

    def __init__(self, **kwargs):
        """
        Args:
            target_size: int, target size
            average_batch: bool, average loss over a batch, default is False
            device: torch.device, device type
        """
        super(CRF, self).__init__()
        for k in kwargs:
            self.__setattr__(k, kwargs[k])
        if not hasattr(self, 'average_batch'):
            self.__setattr__('average_batch', False)
        if not hasattr(self, 'device'):
            self.__setattr__('device',  torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        # init transitions
        self.START_TAG_IDX, self.END_TAG_IDX = -2, -1
        init_transitions = torch.zeros(self.target_size + 2, self.target_size + 2)
        init_transitions[:, self.START_TAG_IDX] = -1000.
        init_transitions[self.END_TAG_IDX, :] = -1000.
        init_transitions = init_transitions.to(self.device)
        self.transitions = nn.Parameter(init_transitions)

    def _forward_alg(self, feats, mask):
        """
        Forward algorithm for computing the partition function (batched)
        in the log space (log Z(x)).

        Args:
            feats: size=(batch_size, seq_len, self.target_size+2)
            mask: size=(batch_size, seq_len)

        Returns:
            summed_log_patitions: 1
            scores: size=(seq_len, batch_size, tag_size, tag_size)
        """
        batch_size = feats.size(0)
        seq_len = feats.size(1)
        tag_size = feats.size(-1)

        mask = mask.transpose(1, 0).contiguous()
        ins_num = batch_size * seq_len

        feats = feats.transpose(1, 0).contiguous().view(
            ins_num, 1, tag_size).expand(ins_num, tag_size, tag_size)

        scores = feats + self.transitions.view(
            1, tag_size, tag_size).expand(ins_num, tag_size, tag_size)
        scores = scores.view(seq_len, batch_size, tag_size, tag_size)

        # s(x, *, y, 1) = s(*, y) + s(x, y)
        partition = scores[0, :, self.START_TAG_IDX, :].clone().view(batch_size, tag_size, 1)

        for idx, cur_values in enumerate(scores[1:]):
            # log a(y, t) = log \sum_{y'} exp (log a(y', t-1) + s(x, y, y', t))
            cur_values = cur_values + partition.contiguous().view(
                batch_size, tag_size, 1).expand(batch_size, tag_size, tag_size)
            cur_partition = log_sum_exp(cur_values, tag_size)

            mask_idx = mask[idx+1, :].view(batch_size, 1).expand(batch_size, tag_size)

            masked_cur_partition = cur_partition.masked_select(mask_idx)
            if masked_cur_partition.dim() != 0:
                mask_idx = mask_idx.contiguous().view(batch_size, tag_size, 1)
                partition.masked_scatter_(mask_idx, masked_cur_partition)

        cur_values = self.transitions.view(1, tag_size, tag_size).expand(
            batch_size, tag_size, tag_size) + partition.contiguous().view(
                batch_size, tag_size, 1).expand(batch_size, tag_size, tag_size)
        cur_partition = log_sum_exp(cur_values, tag_size)
        final_partition = cur_partition[:, self.END_TAG_IDX]
        return final_partition.sum(), scores

    def _viterbi_decode(self, feats, mask):
        """
        Args:
            feats: size=(batch_size, seq_len, self.target_size+2)
            mask: size=(batch_size, seq_len)

        Returns:
            best_path: size=(batch_size, seq_len)
        """
        batch_size = feats.size(0)
        seq_len = feats.size(1)
        tag_size = feats.size(-1)

        length_mask = torch.sum(mask, dim=1).view(batch_size, 1).long()

        mask = mask.transpose(1, 0).contiguous()
        ins_num = seq_len * batch_size

        feats = feats.transpose(1, 0).contiguous().view(
            ins_num, 1, tag_size).expand(ins_num, tag_size, tag_size)

        scores = feats + self.transitions.view(
            1, tag_size, tag_size).expand(ins_num, tag_size, tag_size)
        scores = scores.view(seq_len, batch_size, tag_size, tag_size)

        # record the position of the best score
        back_points, partition_history = [], []

        # mask = 1 + (-1) * mask
        mask = (1 - mask.long()).byte()

        partition = scores[0, :, self.START_TAG_IDX, :].clone().view(batch_size, tag_size, 1)
        partition_history.append(partition)

        for idx, cur_values in enumerate(scores[1:]):
            cur_values = cur_values + partition.contiguous().view(
                batch_size, tag_size, 1).expand(batch_size, tag_size, tag_size)
            partition, cur_bp = torch.max(cur_values, 1)
            partition_history.append(partition.unsqueeze(-1))

            cur_bp.masked_fill_(mask[idx+1].view(batch_size, 1).expand(batch_size, tag_size), 0)
            back_points.append(cur_bp)

        partition_history = torch.cat(partition_history).view(
            seq_len, batch_size, -1).transpose(1, 0).contiguous()

        last_position = length_mask.view(batch_size, 1, 1).expand(batch_size, 1, tag_size) - 1
        last_partition = torch.gather(
            partition_history, 1, last_position).view(batch_size, tag_size, 1)

        last_values = last_partition.expand(batch_size, tag_size, tag_size) + \
            self.transitions.view(1, tag_size, tag_size).expand(batch_size, tag_size, tag_size)
        _, last_bp = torch.max(last_values, 1)
        pad_zero = torch.zeros(batch_size, tag_size).long()
        pad_zero = pad_zero.to(self.device)
        back_points.append(pad_zero)
        back_points = torch.cat(back_points).view(seq_len, batch_size, tag_size)

        pointer = last_bp[:, self.END_TAG_IDX]
        insert_last = pointer.contiguous().view(batch_size, 1, 1).expand(batch_size, 1, tag_size)
        back_points = back_points.transpose(1, 0).contiguous()

        back_points.scatter_(1, last_position, insert_last)

        back_points = back_points.transpose(1, 0).contiguous()

        decode_idx = torch.LongTensor(seq_len, batch_size)
        decode_idx = decode_idx.to(self.device)
        decode_idx[-1] = pointer.data
        for idx in range(len(back_points)-2, -1, -1):
            pointer = torch.gather(back_points[idx], 1, pointer.contiguous().view(batch_size, 1))
            decode_idx[idx] = pointer.view(-1).data
        best_path = decode_idx.transpose(1, 0)
        return best_path, scores

    def forward(self, feats, mask):
        best_path, _ = self._viterbi_decode(feats, mask)
        return best_path

    def _score_sentence(self, scores, mask, tags):
        """
        Args:
            scores: size=(seq_len, batch_size, tag_size, tag_size)
            mask: size=(batch_size, seq_len)
            tags: size=(batch_size, seq_len)

        Returns:
            seq_scores: batch_size
        """
        batch_size = scores.size(1)
        seq_len = scores.size(0)
        tag_size = scores.size(-1)

        # convert tag value into a new format, recorded label bigram information to index
        new_tags = torch.LongTensor(batch_size, seq_len)
        new_tags = new_tags.to(self.device)
        for idx in range(seq_len):
            if idx == 0:
                new_tags[:, 0] = (tag_size - 2) * tag_size + tags[:, 0]
            else:
                new_tags[:, idx] = tags[:, idx-1] * tag_size + tags[:, idx]

        end_transition = self.transitions[:, self.END_TAG_IDX].contiguous().view(
            1, tag_size).expand(batch_size, tag_size)
        length_mask = torch.sum(mask, dim=1).view(batch_size, 1).long()
        end_ids = torch.gather(tags, 1, length_mask-1)
        end_energy = torch.gather(end_transition, 1, end_ids)

        new_tags = new_tags.transpose(1, 0).contiguous().view(seq_len, batch_size, 1)
        tg_energy = torch.gather(scores.view(seq_len, batch_size, -1), 2, new_tags).view(
            seq_len, batch_size)
        tg_energy = tg_energy.masked_select(mask.transpose(1, 0))

        score = tg_energy.sum() + end_energy.sum()

        return score

    def neg_log_likelihood_loss(self, feats, mask, tags):
        """
        Args:
            feats: size=(batch_size, seq_len, tag_size)
            mask: size=(batch_size, seq_len)
            tags: size=(batch_size, seq_len)
        """
        batch_size = feats.size(0)
        forward_score, scores = self._forward_alg(feats, mask)
        gold_score = self._score_sentence(scores, mask, tags)
        if self.average_batch:
            return (forward_score - gold_score) / batch_size
        return forward_score - gold_score

    def ssvm_loss(self, feats, mask, tags, delta=0.25):
        """
        Args:
            feats: size=(batch_size, seq_len, tag_size)
            mask: size=(batch_size, seq_len)
            tags: size=(batch_size, seq_len)
        """
        batch_size = feats.size(0)
        seq_len = feats.size(1)
        _, scores = self._viterbi_decode(feats, mask)
        gold_score = self._score_sentence(scores, mask, tags)

        loss_aug_feats = feats + delta
        offsets = torch.zeros_like(loss_aug_feats)
        offsets.scatter_(2, tags.view(batch_size, seq_len, 1), -delta)
        loss_aug_feats += offsets
        decoded, loss_aug_scores = self._viterbi_decode(loss_aug_feats, mask)
        loss_aug_score = self._score_sentence(loss_aug_scores, mask, decoded)

        if self.average_batch:
            return (loss_aug_score - gold_score) / batch_size
        return los_aug_score - gold_score
