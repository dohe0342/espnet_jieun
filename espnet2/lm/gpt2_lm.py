from typing import Any, List, Tuple

import torch
import torch.nn as nn

from espnet2.lm.abs_model import AbsLM
from espnet.nets.pytorch_backend.transformer.embedding import PositionalEncoding
from espnet.nets.pytorch_backend.transformer.encoder import Encoder
from espnet.nets.pytorch_backend.transformer.mask import subsequent_mask


class GPT2LM(AbsLM):
    def __init__(
        self,
        lm,
        tokenizer,
        device,
        #vocab_size: int,
        #pos_enc: str = None,
        #embed_unit: int = 128,
        #att_unit: int = 256,
        #head: int = 2,
        #unit: int = 1024,
        #layer: int = 4,
        #dropout_rate: float = 0.5,
    ):
        super().__init__()
        self.device = device
        self.lm = lm
        self.tokenizer = tokenizer
        self.log_softmax = torch.nn.LogSoftmax(dim=-1)

        '''
        self.int_am2lm_dict = {
                             0:95,
                             1:95,
                             2:94,    #|
                             3:36,    #E
                             4:51,    #T
                             5:32,    #A
                             6:46,    #O
                             7:45,    #N
                             8:40,    #I
                             9:39,   #H
                             10:50,   #S
                             11:49,   #R
                             12:35,   #D
                             13:43,   #L
                             14:52,   #U
                             15:44,   #M
                             16:54,   #W
                             17:34,   #C
                             18:37,   #F
                             19:38,   #G
                             20:56,   #Y
                             21:47,   #P
                             22:33,   #B
                             23:53,   #V
                             24:42,   #K
                             25:6,    #'
                             26:55,   #X
                             27:41,   #J
                             28:48,   #Q
                             29:57,   #Z
                             30:95,
                         }
        
        self.convert_matrix = torch.zeros(96, 31).to(device)
        self.convert_matrix[95][0] = 1
        self.convert_matrix[95][1] = 1
        self.convert_matrix[94][2] = 1
        self.convert_matrix[36][3] = 1
        self.convert_matrix[51][4] = 1
        self.convert_matrix[32][5] = 1
        self.convert_matrix[46][6] = 1
        self.convert_matrix[45][7] = 1
        self.convert_matrix[40][8] = 1
        self.convert_matrix[39][9] = 1
        self.convert_matrix[50][10] = 1
        self.convert_matrix[49][11] = 1
        self.convert_matrix[35][12] = 1
        self.convert_matrix[43][13] = 1
        self.convert_matrix[52][14] = 1
        self.convert_matrix[44][15] = 1
        self.convert_matrix[54][16] = 1
        self.convert_matrix[34][17] = 1
        self.convert_matrix[37][18] = 1
        self.convert_matrix[38][19] = 1
        self.convert_matrix[56][20] = 1
        self.convert_matrix[47][21] = 1
        self.convert_matrix[33][22] = 1
        self.convert_matrix[53][23] = 1
        self.convert_matrix[42][24] = 1
        self.convert_matrix[6][25] = 1
        self.convert_matrix[55][26] = 1
        self.convert_matrix[41][27] = 1
        self.convert_matrix[48][28] = 1
        self.convert_matrix[57][29] = 1
        self.convert_matrix[95][30] = 1
        '''

        self.int_am2lm_dict = {
                             0:95,
                             1:95,
                             2:95,
                             3:95,
                             4:94,    #|
                             5:36,    #E
                             6:51,    #T
                             7:32,    #A
                             8:46,    #O
                             9:45,    #N
                             10:40,   #I
                             11:39,   #H
                             12:50,   #S
                             13:49,   #R
                             14:35,   #D
                             15:43,   #L
                             16:52,   #U
                             17:44,   #M
                             18:34,   #C
                             19:54,   #W
                             20:37,   #F
                             21:38,   #G
                             22:56,   #Y
                             23:47,   #P
                             24:33,   #B
                             25:53,   #V
                             26:42,   #K
                             27:6,    #'
                             28:55,   #X
                             29:41,   #J
                             30:48,   #Q
                             31:57,   #Z
                         }
        
        self.convert_matrix = torch.zeros(96, 32).to(device)
        self.convert_matrix[95][0] = 1
        self.convert_matrix[95][1] = 1
        self.convert_matrix[95][2] = 1
        self.convert_matrix[95][3] = 1
        self.convert_matrix[94][4] = 1
        self.convert_matrix[36][5] = 1
        self.convert_matrix[51][6] = 1
        self.convert_matrix[32][7] = 1
        self.convert_matrix[46][8] = 1
        self.convert_matrix[45][9] = 1
        self.convert_matrix[40][10] = 1
        self.convert_matrix[39][11] = 1
        self.convert_matrix[50][12] = 1
        self.convert_matrix[49][13] = 1
        self.convert_matrix[35][14] = 1
        self.convert_matrix[43][15] = 1
        self.convert_matrix[52][16] = 1
        self.convert_matrix[44][17] = 1
        self.convert_matrix[34][18] = 1
        self.convert_matrix[54][19] = 1
        self.convert_matrix[37][20] = 1
        self.convert_matrix[38][21] = 1
        self.convert_matrix[56][22] = 1
        self.convert_matrix[47][23] = 1
        self.convert_matrix[33][24] = 1
        self.convert_matrix[53][25] = 1
        self.convert_matrix[42][26] = 1
        self.convert_matrix[6][27] = 1
        self.convert_matrix[55][28] = 1
        self.convert_matrix[41][29] = 1
        self.convert_matrix[48][30] = 1
        self.convert_matrix[57][31] = 1

        print(self.convert_matrix)

    def _target_mask(self, ys_in_pad):
        ys_mask = ys_in_pad != 0
        m = subsequent_mask(ys_mask.size(-1), device=ys_mask.device).unsqueeze(0)
        return ys_mask.unsqueeze(-2) & m

    def forward(self, input: torch.Tensor, hidden: None) -> Tuple[torch.Tensor, None]:
        """Compute LM loss value from buffer sequences.

        Args:
            input (torch.Tensor): Input ids. (batch, len)
            hidden (torch.Tensor): Target ids. (batch, len)

        """
        x = self.embed(input)
        mask = self._target_mask(input)
        h, _ = self.encoder(x, mask)
        y = self.decoder(h)
        return y, None

    def score(
        self, y: torch.Tensor, state: Any, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Any]:
        """Score new token.

        Args:
            y (torch.Tensor): 1D torch.int64 prefix tokens.
            state: Scorer state for prefix tokens
            x (torch.Tensor): encoder feature that generates ys.

        Returns:
            tuple[torch.Tensor, Any]: Tuple of
                torch.float32 scores for next token (vocab_size)
                and next state for ys

        """
        
        y = y.unsqueeze(0)
        h, _, cache = self.encoder.forward_one_step(
            self.embed(y), self._target_mask(y), cache=state
        )
        h = self.decoder(h[:, -1])
        logp = h.log_softmax(dim=-1).squeeze(0)
        return logp, cache

    def batch_score(
        self, ys: torch.Tensor, states: List[Any], xs: torch.Tensor
    ) -> Tuple[torch.Tensor, List[Any]]:
        
        ys_new = []
        for i, b in enumerate(ys):
            ys_new.append([])
            for tensor in b:
                ys_new[i].append(self.int_am2lm_dict[tensor.item()])
        
        ys_new = torch.LongTensor(ys_new).to(ys.device)
        n_batch = len(ys_new)
        n_layers = 12
        h = self.lm(ys_new)
        h, states = h['logits'], h['past_key_values']
        h = h[:,-1]
        h = torch.matmul(h, self.convert_matrix)
        
        logp = self.log_softmax(h)

        #print(len(states))
        #print(len(states[0]))

        # transpose state of [layer, batch] into [batch, layer]
        #state_list = [[states[i][b] for i in range(n_layers)] for b in range(n_batch)]
        state_list = None
        return logp, state_list
