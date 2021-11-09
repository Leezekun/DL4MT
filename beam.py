import copy
from heapq import heappush, heappop

import torch


# BeamSearchNode {{{
class BeamSearchNode(object):
    def __init__(self, h1, c1, h2, c2, input_feed, prev_node, wid, logp, length):
        self.h1 = h1
        self.c1 = c1
        self.h2 = h2
        self.c2 = c2
        self.input_feed = input_feed
        self.prev_node = prev_node
        self.wid = wid
        self.logp = logp
        self.length = length

    def eval(self):
        return self.logp / float(self.length - 1 + 1e-6)
# }}}

# beam search {{{
def beam_search_decoding(decoder,
                         enc_outs,
                         enc_masks,
                         dec_hiddens1,
                         dec_cells1,
                         dec_hiddens2,
                         dec_cells2,
                         dec_input_feeds,
                         embeddings,
                         beam_width,
                         n_best,
                         sos_token,
                         eos_token,
                         max_dec_steps,
                         device):
    """Beam Seach Decoding for HybridNMT

    Args:
        decoder: An LSTM decoder model
        enc_outs: A sequence of encoded input. (trg_len, batch_size, enc_out_dim). 
        beam_width: Beam search width
        n_best: The number of output sequences for each input

    Returns:
        n_best_list: Decoded N-best results. (bs, T)
    """

    assert beam_width >= n_best

    n_best_list = []
    bs = enc_outs.shape[1]

    # Decoding goes sentence by sentence.
    # So this process is very slow compared to batch decoding process.
    for batch_id in range(bs):
        # Get last encoder hidden state
        decoder_hidden1 = dec_hiddens1[batch_id]
        decoder_cell1 = dec_cells1[batch_id]
        decoder_hidden2 = dec_hiddens2[batch_id]
        decoder_cell2 = dec_cells2[batch_id]
        decoder_input_feed = dec_input_feeds[batch_id]
        enc_out = enc_outs[:, batch_id, :] # (trg_len, enc_out_dim)
        enc_mask = enc_masks[batch_id]

        # Prepare first token for decoder
        decoder_input = torch.tensor([sos_token]).long().to(device) # (1)

        # Number of sentence to generate
        end_nodes = []

        # starting node
        node = BeamSearchNode(h1=decoder_hidden1, c1=decoder_cell1, h2=decoder_hidden2, c2=decoder_cell2, input_feed=decoder_input_feed, prev_node=None, wid=decoder_input, logp=0, length=1)

        # whole beam search node graph
        nodes = []

        # Start the queue
        heappush(nodes, (-node.eval(), id(node), node))
        n_dec_steps = 0

        # Start beam search
        while True:
            # Give up when decoding takes too long
            if n_dec_steps > max_dec_steps:
                break

            # Fetch the best node
            score, _, n = heappop(nodes)
            decoder_input = embeddings(n.wid)
            decoder_hidden1 = n.h1
            decoder_cell1 = n.c1
            decoder_hidden2 = n.h2
            decoder_cell2 = n.c2
            decoder_input_feed = n.input_feed 

            if n.wid.item() == eos_token and n.prev_node is not None:
                end_nodes.append((score, id(n), n))
                # If we reached maximum # of sentences required
                if len(end_nodes) >= n_best:
                    break
                else:
                    continue

            # Decode for one step using decoder
            decoder_output, decoder_hidden1, decoder_cell1, decoder_hidden2, decoder_cell2, decoder_input_feed = decoder(decoder_input, decoder_input_feed.unsqueeze(0), 
                                                                                                                        decoder_hidden1.unsqueeze(0), decoder_cell1.unsqueeze(0), 
                                                                                                                        decoder_hidden2.unsqueeze(0), decoder_cell2.unsqueeze(0), 
                                                                                                                        enc_out.unsqueeze(1), enc_mask.unsqueeze(0))
                                                                                                                        
            # Get top-k from this decoded result
            topk_log_prob, topk_indexes = torch.topk(decoder_output, beam_width) # (1, bw), (1, bw)
            # Then, register new top-k nodes
            for new_k in range(beam_width):
                decoded_t = topk_indexes[0][new_k].view(1) # (1)
                logp = topk_log_prob[0][new_k].item() # float log probability val

                node = BeamSearchNode(h1=decoder_hidden1.squeeze(0),
                                      c1=decoder_cell1.squeeze(0),
                                      h2=decoder_hidden2.squeeze(0),
                                      c2=decoder_cell2.squeeze(0),
                                      input_feed=decoder_input_feed.squeeze(0),
                                      prev_node=n,
                                      wid=decoded_t,
                                      logp=n.logp+logp,
                                      length=n.length+1)
                heappush(nodes, (-node.eval(), id(node), node))
            n_dec_steps += beam_width

        # if there are no end_nodes, retrieve best nodes (they are probably truncated)
        if len(end_nodes) == 0:
            end_nodes = [heappop(nodes) for _ in range(beam_width)]

        # Construct sequences from end_nodes
        n_best_seq_list = []
        for score, _id, n in sorted(end_nodes, key=lambda x: x[0]):
            sequence = [n.wid.item()]
            # back trace from end node
            while n.prev_node is not None:
                n = n.prev_node
                sequence.append(n.wid.item())
            sequence = sequence[::-1] # reverse

            n_best_seq_list.append(sequence)

        n_best_list.append(n_best_seq_list)

    return n_best_list
# }}}

# batch beam search {{{
def batch_beam_search_decoding(decoder,
                                enc_outs,
                                enc_masks,
                                dec_hiddens1,
                                dec_cells1,
                                dec_hiddens2,
                                dec_cells2,
                                dec_input_feeds,
                                embeddings,
                                beam_width,
                                n_best,
                                sos_token,
                                eos_token,
                                max_dec_steps,
                                device):
    """Batch Beam Seach Decoding for RNN

    Args:
        decoder: An RNN decoder model
        enc_outs: A sequence of encoded input. (T, bs, 2H). 2H for bidirectional
        enc_last_h: (bs, H)
        beam_width: Beam search width
        n_best: The number of output sequences for each input

    Returns:
        n_best_list: Decoded N-best results. (bs, T)
    """

    assert beam_width >= n_best

    n_best_list = []
    bs = enc_outs.shape[1]

    # Prepare first token for decoder
    decoder_input = torch.tensor([sos_token]).repeat(1, bs).long().to(device) # (1, bs)

    # Number of sentence to generate
    end_nodes_list = [[] for _ in range(bs)]

    # whole beam search node graph
    nodes = [[] for _ in range(bs)]

    # Start the queue
    for batch_id in range(bs):
        decoder_hidden1 = dec_hiddens1[batch_id] #[dec_hid_dim]
        decoder_cell1 = dec_cells1[batch_id] #[dec_hid_dim]
        decoder_hidden2 = dec_hiddens2[batch_id] #[dec_hid_dim]
        decoder_cell2 = dec_cells2[batch_id] #[dec_hid_dim]
        decoder_input_feed = dec_input_feeds[batch_id] #[dec_out_dim]

        # starting node
        node = BeamSearchNode(h1=decoder_hidden1, c1=decoder_cell1, h2=decoder_hidden2, c2=decoder_cell2, input_feed=decoder_input_feed, prev_node=None, wid=decoder_input[:, batch_id], logp=0, length=1)
        heappush(nodes[batch_id], (-node.eval(), id(node), node))

    # Start beam search
    fin_nodes = set()
    history = [None for _ in range(bs)]
    n_dec_steps_list = [0 for _ in range(bs)]
    while len(fin_nodes) < bs:
        # Fetch the best node
        batch_decoder_input, batch_decoder_hidden1, batch_decoder_cell1, batch_decoder_hidden2, batch_decoder_cell2, batch_decoder_input_feed = [], [], [], [], [], []
        for bid in range(bs):
            if bid not in fin_nodes and n_dec_steps_list[bid] > max_dec_steps:
                fin_nodes.add(bid)

            if bid in fin_nodes:
                score, n = history[bid] # dummy for data consistency
            else:
                score, _, n = heappop(nodes[bid])
                if n.wid.item() == eos_token and n.prev_node is not None:
                    end_nodes_list[bid].append((score, id(n), n))
                    # If we reached maximum # of sentences required
                    if len(end_nodes_list[bid]) >= n_best:
                        fin_nodes.add(bid)
                history[bid] = (score, n)
            batch_decoder_input.append(embeddings(n.wid))
            batch_decoder_hidden1.append(n.h1)
            batch_decoder_cell1.append(n.c1)
            batch_decoder_hidden2.append(n.h2)
            batch_decoder_cell2.append(n.c2)
            batch_decoder_input_feed.append(n.input_feed)

        batch_decoder_input = torch.cat(batch_decoder_input).to(device) # (bs, emb_size)
        batch_decoder_hidden1 = torch.stack(batch_decoder_hidden1, 0).to(device) # (bs, H)
        batch_decoder_cell1 = torch.stack(batch_decoder_cell1, 0).to(device) # (bs, H)
        batch_decoder_hidden2 = torch.stack(batch_decoder_hidden2, 0).to(device) # (bs, H)
        batch_decoder_cell2 = torch.stack(batch_decoder_cell2, 0).to(device) # (bs, H)
        batch_decoder_input_feed = torch.stack(batch_decoder_input_feed, 0).to(device) # (bs, H)

        # Decode for one step using decoder
        batch_decoder_output, batch_decoder_hidden1, batch_decoder_cell1, batch_decoder_hidden2, batch_decoder_cell2, batch_decoder_input_feed = decoder(batch_decoder_input, batch_decoder_input_feed, 
                                                                                                                        batch_decoder_hidden1, batch_decoder_cell1, 
                                                                                                                        batch_decoder_hidden2, batch_decoder_cell2, 
                                                                                                                        enc_outs, enc_masks)
        # Get top-k from this decoded result
        topk_log_prob, topk_indexes = torch.topk(batch_decoder_output, beam_width) # (bs, bw), (bs, bw)
        # Then, register new top-k nodes
        for bid in range(bs):
            if bid in fin_nodes:
                continue
            score, n = history[bid]
            if n.wid.item() == eos_token and n.prev_node is not None:
                continue
            for new_k in range(beam_width):
                decoded_t = topk_indexes[bid][new_k].view(1) # (1)
                logp = topk_log_prob[bid][new_k].item() # float log probability val

                node = BeamSearchNode(h1=batch_decoder_hidden1[bid], c1=batch_decoder_cell1[bid], 
                                      h2=batch_decoder_hidden2[bid], c2=batch_decoder_cell2[bid],
                                      input_feed=batch_decoder_input_feed[bid],
                                      prev_node=n,
                                      wid=decoded_t,
                                      logp=n.logp+logp,
                                      length=n.length+1)
                heappush(nodes[bid], (-node.eval(), id(node), node))
            n_dec_steps_list[bid] += beam_width

    # Construct sequences from end_nodes
    # if there are no end_nodes, retrieve best nodes (they are probably truncated)
    for bid in range(bs):
        if len(end_nodes_list[bid]) == 0:
            end_nodes_list[bid] = [heappop(nodes[bid]) for _ in range(beam_width)]

        n_best_seq_list = []
        for score, _id, n in sorted(end_nodes_list[bid], key=lambda x: x[0]):
            sequence = [n.wid.item()]
            while n.prev_node is not None:
                n = n.prev_node
                sequence.append(n.wid.item())
            sequence = sequence[::-1] # reverse

            n_best_seq_list.append(sequence)

        n_best_list.append(copy.copy(n_best_seq_list))

    return n_best_list
# }}}

