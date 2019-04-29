import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.dart_genotypes import PRIMITIVES, STEPS, CONCAT, Genotype
from torch.autograd import Variable
from collections import namedtuple
from models.dart_model import DARTSCell, RNNModel


class DARTSCellSearch(DARTSCell):

  def __init__(self, embedding_size, hidden_size, dropouth, dropoutx):
    super(DARTSCellSearch, self).__init__(embedding_size, hidden_size, dropouth, dropoutx, genotype=None)
    self.bn = nn.BatchNorm1d(hidden_size, affine=False).cuda()

  def cell(self, x, h_prev, x_mask, h_mask):
    s0 = self._compute_init_state(x, h_prev, x_mask, h_mask)
    s0 = self.bn(s0) # batch normalization
    probs = F.softmax(self.weights, dim=-1)

    offset = 0
    states = s0.unsqueeze(0)
    for i in range(STEPS):
      if self.training:
        masked_states = states * h_mask.unsqueeze(0)
      else:
        masked_states = states
      ch = masked_states.view(-1, self.hidden_size).mm(self._Ws[i]).view(i+1, -1, 2*self.hidden_size)
      c, h = torch.split(ch, self.hidden_size, dim=-1)
      c = c.sigmoid()

      s = torch.zeros_like(s0)
      for k, name in enumerate(PRIMITIVES):
        if name == 'none': # no activation function, skip
          continue
        else: # has activation function
          fn = self._get_activation(name)
          unweighted = states + c * (fn(h) - states)
          s += torch.sum(probs[offset:offset+i+1, k].unsqueeze(-1).unsqueeze(-1) * unweighted, dim=0)
      s = self.bn(s) # batch_normalization
      states = torch.cat([states, s.unsqueeze(0)], 0)
      offset += i+1
    output = torch.mean(states[-CONCAT:], dim=0)
    return output


class RNNModelSearch(RNNModel):

    def __init__(self, *args):
        super(RNNModelSearch, self).__init__(*args, cell_cls=DARTSCellSearch, genotype=None)
        self._args = args
        self._initialize_arch_parameters()

    def new(self):
        model_new = RNNModelSearch(*self._args)
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def _initialize_arch_parameters(self):
      k = sum(i for i in range(1, STEPS+1))
      weights_data = torch.randn(k, len(PRIMITIVES)).mul_(1e-3) # weights data, unknown?????
      self.weights = Variable(weights_data.cuda(), requires_grad=True)
      self._arch_parameters = [self.weights]
      for rnn in self.rnns:
        rnn.weights = self.weights

    def arch_parameters(self):
      return self._arch_parameters

    def _loss(self, hidden, input, target):
      # print("hohoh", hidden)
      prob, hidden_next = self(input, hidden)
      criterion = nn.CrossEntropyLoss()
      loss = criterion(prob.squeeze(), target)
      # loss = nn.functional.nll_loss(log_prob.view(-1, log_prob.size(2)), target)
      return loss, hidden_next

    def genotype(self):

      def _parse(probs):
        gene = []
        start = 0
        for i in range(STEPS):
          end = start + i + 1
          W = probs[start:end].copy()
          j = sorted(range(i + 1), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[0]
          k_best = None
          for k in range(len(W[j])):
            if k != PRIMITIVES.index('none'):
              if k_best is None or W[j][k] > W[j][k_best]:
                k_best = k
          gene.append((PRIMITIVES[k_best], j))
          start = end
        return gene

      gene = _parse(F.softmax(self.weights, dim=-1).data.cpu().numpy())
      genotype = Genotype(recurrent=gene, concat=range(STEPS+1)[-CONCAT:])
      return genotype

